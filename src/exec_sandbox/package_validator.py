"""Package validation for code execution safety.

Package catalogs built at development time (make upgrade → make build-catalogs).
No runtime network calls, catalogs bundled in container.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from exec_sandbox.exceptions import ErrorCode, InvalidInputError
from exec_sandbox.models import Language  # noqa: TC001 - Used at runtime (language.value)

# Package name extraction pattern
# Matches: alphanumeric, underscore, hyphen, dot followed by version operator
# Python: ==, ~=, >=, <=, !=, >, <
# JavaScript: @
_PACKAGE_NAME_PATTERN = re.compile(r"^([a-zA-Z0-9_\-\.]+)[@=<>~]")


class PackageValidator:
    """Validates packages against bundled allow-lists.

    Security mechanism to prevent execution of arbitrary packages.
    Only packages from curated allow-lists are permitted.

    Version validation is enforced at schema level via Pydantic.
    This validator only checks allow-list membership.

    Catalogs are built at dev time and bundled in container (no runtime fetching).

    Use the async factory method `create()` to instantiate. It parses about
    400 KB of JSON: create one validator and reuse it.
    """

    def __init__(self, allow_lists: dict[str, set[str]]):
        """Private constructor - use create() instead.

        Args:
            allow_lists: Pre-loaded allow-lists dict mapping language to package sets
        """
        self._allow_lists = allow_lists

    @classmethod
    async def create(
        cls,
        pypi_allow_list_path: Path | None = None,
        npm_allow_list_path: Path | None = None,
    ) -> PackageValidator:
        """Async factory method to create a PackageValidator.

        Args:
            pypi_allow_list_path: Path to JSON file with PyPI package names (bundled).
                Defaults to bundled resources/pypi_top_10k.json
            npm_allow_list_path: Path to JSON file with npm package names (bundled).
                Defaults to bundled resources/npm_top_10k.json

        Returns:
            Initialized PackageValidator instance
        """
        catalogs_dir = Path(__file__).parent / "resources"
        pypi_path = pypi_allow_list_path or catalogs_dir / "pypi_top_10k.json"
        npm_path = npm_allow_list_path or catalogs_dir / "npm_top_10k.json"

        # One worker-thread job: reading and parsing both files takes about 2 ms
        return cls(await asyncio.to_thread(cls._load_allow_lists, pypi_path, npm_path))

    @classmethod
    def _load_allow_lists(cls, pypi_path: Path, npm_path: Path) -> dict[str, set[str]]:
        return {"python": cls._load_allow_list(pypi_path), "javascript": cls._load_allow_list(npm_path)}

    @staticmethod
    def _load_allow_list(path: Path) -> set[str]:
        """Load allow-list from bundled JSON file.

        Args:
            path: Path to JSON file containing list of package names

        Returns:
            Set of allowed package names (lowercase for case-insensitive matching)
        """
        packages: list[str] = json.loads(path.read_text())
        return {pkg.lower() for pkg in packages}

    def validate(self, packages: list[str], language: Language) -> None:
        """Validate packages against allow-list.

        Version validation is handled at schema level.

        Args:
            packages: List of package specifiers (e.g., ["pandas==2.0.0", "lodash@4.17.21"])
            language: Programming language to validate for

        Raises:
            InvalidInputError: PACKAGE_NOT_ALLOWED if a package is not in the allow-list;
                ARGUMENT_INVALID if a spec has no version or the language has no packages
        """
        if language.value not in self._allow_lists:
            raise InvalidInputError(
                f"Package validation not supported for '{language.value}' language. "
                f"Supported: {', '.join(sorted(self._allow_lists))}.",
                code=ErrorCode.ARGUMENT_INVALID,
                context=None,
                hint=None,
                instance_id=None,
            )

        allow_list = self._allow_lists[language.value]

        for package_spec in packages:
            match = _PACKAGE_NAME_PATTERN.match(package_spec)
            if not match:
                # Should never happen due to schema validation, but defensive
                raise InvalidInputError(
                    f"Invalid package spec: '{package_spec}'. "
                    f"Must include version specifier (e.g., pandas==2.0.0 or lodash@4.17.21).",
                    code=ErrorCode.ARGUMENT_INVALID,
                    context=None,
                    hint=None,
                    instance_id=None,
                )

            base_name = match.group(1).strip().lower()

            # Check if package is in allow-list
            if base_name not in allow_list:
                raise InvalidInputError(
                    f"Package '{base_name}' not in {language.value} allow-list. "
                    f"Only pre-approved packages from the catalog are permitted.",
                    code=ErrorCode.PACKAGE_NOT_ALLOWED,
                    context=None,
                    hint=None,
                    instance_id=None,
                )

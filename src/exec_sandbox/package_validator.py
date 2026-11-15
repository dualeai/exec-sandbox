"""Package validation for code execution safety.

Package catalogs built at development time (make upgrade → make build-catalogs)
and shipped in the package (resources/). No network call at run time.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Final

from exec_sandbox.exceptions import ErrorCode, InvalidInputError
from exec_sandbox.models import Language  # noqa: TC001 - Used at runtime (language.value)

# Package name extraction pattern
# Matches: alphanumeric, underscore, hyphen, dot followed by the first character of a version operator
# Python: ==, ~=, >=, <=, >, < (not !=: "!" is not a name character, so no match)
# JavaScript: @
_PACKAGE_NAME_PATTERN = re.compile(r"^([a-zA-Z0-9_\-\.]+)[@=<>~]")


def _load_allow_lists(paths: dict[str, Path]) -> dict[str, set[str]]:
    """The allow-list of each language value from its JSON file (a list of package names),
    lowercase for case-insensitive matching. Blocking: create() runs it in a worker thread."""
    return {language: {pkg.lower() for pkg in json.loads(path.read_text())} for language, path in paths.items()}


BUNDLED_PYPI_ALLOW_LIST: Final = Path(__file__).parent / "resources" / "pypi_top_10k.json"
"""The PyPI allow-list shipped in the package (make build-catalogs)."""
BUNDLED_NPM_ALLOW_LIST: Final = Path(__file__).parent / "resources" / "npm_top_10k.json"
"""The npm allow-list shipped in the package (make build-catalogs)."""


class PackageValidator:
    """Validates packages against bundled allow-lists.

    Security mechanism to prevent execution of arbitrary packages.
    Only packages from curated allow-lists are permitted.

    It also refuses a spec with no version operator (_PACKAGE_NAME_PATTERN). It
    does not check the version itself: the guest agent checks the whole spec.

    Catalogs are built at dev time and shipped in the package (no runtime fetching).

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
    async def create(cls, *, pypi_allow_list_path: Path, npm_allow_list_path: Path) -> PackageValidator:
        """Async factory method to create a PackageValidator.

        Args:
            pypi_allow_list_path: Path to JSON file with PyPI package names (BUNDLED_PYPI_ALLOW_LIST)
            npm_allow_list_path: Path to JSON file with npm package names (BUNDLED_NPM_ALLOW_LIST)

        Returns:
            Initialized PackageValidator instance
        """
        # One worker-thread job: reading and parsing both files takes about 2 ms
        return cls(
            await asyncio.to_thread(
                _load_allow_lists, {"python": pypi_allow_list_path, "javascript": npm_allow_list_path}
            )
        )

    def validate(self, packages: list[str], language: Language) -> None:
        """Validate packages against allow-list.

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
                # No version operator after the name (VmSpec does not check the specs)
                raise InvalidInputError(
                    f"Invalid package spec: '{package_spec}'. "
                    f"Must include version specifier (e.g., pandas==2.0.0 or lodash@4.17.21).",
                    code=ErrorCode.ARGUMENT_INVALID,
                    context=None,
                    hint=None,
                    instance_id=None,
                )

            base_name = match.group(1).lower()

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

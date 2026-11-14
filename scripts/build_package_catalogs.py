#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["aiohttp", "tenacity"]
# ///
"""Build package allow-lists from PyPI and npm registries.

Fetches the top N packages by download count to make security allow-lists.
`make upgrade` runs it (target build-catalogs), which writes
pypi_top_10k.json and npm_top_10k.json to src/exec_sandbox/resources.
"""

import asyncio
import json
from pathlib import Path

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Packages kept per registry (the file names say 10k).
TOP_N = 10_000


async def fetch_pypi_top_packages() -> list[str]:
    """Fetch the TOP_N PyPI packages by download count.

    Uses the hugovk/top-pypi-packages list (downloads of the last 30 days,
    from the PyPI BigQuery dataset).

    Returns:
        List of package names sorted by download count
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def fetch_pypi_data() -> dict[str, list[dict[str, str]]]:
        """Fetch PyPI top packages with retry on transient errors."""
        url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
        async with (
            aiohttp.ClientSession() as session,
            session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp,
        ):
            resp.raise_for_status()
            return await resp.json()

    data = await fetch_pypi_data()

    # Extract package names from rows
    return [row["project"] for row in data["rows"][:TOP_N]]


async def fetch_npm_top_packages() -> list[str]:
    """Fetch the TOP_N npm packages from static GitHub list.

    Uses evanwashere/top-npm-packages which ranks packages by monthly downloads.
    npm's search API has restrictions (no single-letter searches, rate limits, complex pagination).
    Static list is simpler and sufficient for security allow-list (rarely changes).

    Returns:
        List of package names sorted by download popularity (most popular first)
    """
    url = "https://raw.githubusercontent.com/evanwashere/top-npm-packages/master/all.json"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def fetch_list() -> list[str]:
        """Fetch package list with retry."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp,
        ):
            resp.raise_for_status()
            text = await resp.text()

        # Parse JSON from text (GitHub serves as text/plain)
        data: list[list[str | int]] = json.loads(text)

        # Extract package names from format: [["package-name", downloads], ...]
        # Type guard: data is already typed as list[list[str | int]] from the annotation
        # We check structure for runtime safety, not for type narrowing
        if data and len(data[0]) > 0:
            return [str(pkg[0]) for pkg in data if len(pkg) > 0]
        return []

    packages = await fetch_list()
    return packages[:TOP_N]


async def build_catalogs(output_dir: Path) -> None:
    """Build package catalogs for PyPI and npm.

    Args:
        output_dir: Directory to write JSON catalog files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Fetching top {TOP_N} PyPI packages...")
    print(f"📦 Fetching top {TOP_N} npm packages...")

    # Fetch in parallel
    pypi_packages, npm_packages = await asyncio.gather(
        fetch_pypi_top_packages(),
        fetch_npm_top_packages(),
    )

    # Write catalogs
    pypi_path = output_dir / "pypi_top_10k.json"
    with pypi_path.open("w") as f:
        json.dump(pypi_packages, f, indent=2)
    print(f"✅ Wrote {len(pypi_packages)} PyPI packages to {pypi_path}")

    npm_path = output_dir / "npm_top_10k.json"
    with npm_path.open("w") as f:
        json.dump(npm_packages, f, indent=2)
    print(f"✅ Wrote {len(npm_packages)} npm packages to {npm_path}")


if __name__ == "__main__":
    import sys

    # Get output directory from args or use default
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("catalogs")

    asyncio.run(build_catalogs(output_dir))

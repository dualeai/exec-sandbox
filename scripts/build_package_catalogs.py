#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "aiohttp~=3.13",
#   "tenacity~=9.1",
# ]
# ///
"""Build package allow-lists from PyPI and npm registries.

Fetches top N packages by download count to create security allow-lists.
Run during `make upgrade` to keep catalogs fresh with dependencies.
"""

import asyncio
import json
from pathlib import Path

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


async def fetch_pypi_top_packages(limit: int = 10_000) -> list[str]:
    """Fetch top PyPI packages by download count.

    Uses pypistats.org API (BigQuery data, updated daily).

    Args:
        limit: Number of top packages to fetch

    Returns:
        List of package names sorted by download count
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def fetch_pypi_data() -> dict:
        """Fetch PyPI top packages with retry on transient errors."""
        url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                return await resp.json()

    data = await fetch_pypi_data()

    # Extract package names from rows
    packages = [row["project"] for row in data["rows"][:limit]]
    return packages


async def fetch_npm_top_packages(limit: int = 10_000) -> list[str]:
    """Fetch top npm packages from static GitHub list.

    Uses evanwashere/top-npm-packages which ranks packages by monthly downloads.
    npm's search API has restrictions (no single-letter searches, rate limits, complex pagination).
    Static list is simpler and sufficient for security allow-list (rarely changes).

    Args:
        limit: Number of top packages to fetch

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
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                text = await resp.text()

        # Parse JSON from text (GitHub serves as text/plain)
        data = json.loads(text)

        # Extract package names from format: [["package-name", downloads], ...]
        if isinstance(data, list) and data and isinstance(data[0], list):
            return [pkg[0] for pkg in data if len(pkg) > 0]
        return []

    packages = await fetch_list()
    return packages[:limit]


async def build_catalogs(output_dir: Path, pypi_limit: int = 10_000, npm_limit: int = 10_000) -> None:
    """Build package catalogs for PyPI and npm.

    Args:
        output_dir: Directory to write JSON catalog files
        pypi_limit: Number of top PyPI packages
        npm_limit: Number of top npm packages
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Fetching top {pypi_limit} PyPI packages...")
    print(f"📦 Fetching top {npm_limit} npm packages...")

    # Fetch in parallel
    pypi_packages, npm_packages = await asyncio.gather(
        fetch_pypi_top_packages(pypi_limit),
        fetch_npm_top_packages(npm_limit),
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

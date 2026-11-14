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
import sys
from pathlib import Path
from typing import Any

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Packages kept per registry (the file names say 10k).
TOP_N = 10_000

# hugovk/top-pypi-packages: downloads of the last 30 days, from the PyPI BigQuery dataset.
PYPI_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
# evanwashere/top-npm-packages: ranked by monthly downloads. npm's search API has
# restrictions (no single-letter searches, rate limits, complex pagination).
NPM_URL = "https://raw.githubusercontent.com/evanwashere/top-npm-packages/master/all.json"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError),
)
async def fetch_json(url: str) -> Any:
    """GET a JSON document, with retry on transient errors (GitHub serves some as text/plain)."""
    async with (
        aiohttp.ClientSession() as session,
        session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp,
    ):
        resp.raise_for_status()
        return json.loads(await resp.text())


async def fetch_pypi_top_packages() -> list[str]:
    """The TOP_N PyPI package names, most downloaded first (rows: {"project": name, ...})."""
    data = await fetch_json(PYPI_URL)
    return [row["project"] for row in data["rows"][:TOP_N]]


async def fetch_npm_top_packages() -> list[str]:
    """The TOP_N npm package names, most downloaded first (rows: [name, downloads])."""
    data = await fetch_json(NPM_URL)
    return [str(row[0]) for row in data[:TOP_N]]


async def build_catalogs(output_dir: Path) -> None:
    """Write pypi_top_10k.json and npm_top_10k.json to output_dir."""
    print(f"📦 Fetching top {TOP_N} PyPI and npm packages...")
    pypi_packages, npm_packages = await asyncio.gather(fetch_pypi_top_packages(), fetch_npm_top_packages())

    for registry, packages in (("pypi", pypi_packages), ("npm", npm_packages)):
        path = output_dir / f"{registry}_top_10k.json"
        path.write_text(json.dumps(packages, indent=2))
        print(f"✅ Wrote {len(packages)} {registry} packages to {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:  # the script name and the output folder
        sys.exit(f"Usage: {sys.argv[0]} <output folder>  (make build-catalogs passes src/exec_sandbox/resources)")
    asyncio.run(build_catalogs(Path(sys.argv[1])))

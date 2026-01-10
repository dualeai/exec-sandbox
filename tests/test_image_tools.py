"""Tests for image tool availability.

Verifies each image has the expected tools installed and working.
Uses a parameterized matrix for easy maintenance.
"""

from pathlib import Path

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler

# Images directory - relative to repo root
images_dir = Path(__file__).parent.parent / "images" / "dist"


# =============================================================================
# Tool availability tests using Language.RAW (shell commands)
# =============================================================================
# Note: Language.RAW uses the raw image. To test tools on Python/JS images,
# we use language-native code (import subprocess, child_process, etc.)

# Common tools to test on the RAW image
RAW_IMAGE_TOOLS = [
    ("curl", "curl --version"),
    ("git", "git --version"),
    ("jq", "jq --version"),
    ("bash", "bash --version"),
    ("tar", "tar --version"),
    ("gzip", "gzip --version"),
    ("unzip", "unzip -v"),
    ("file", "file --version"),
]


@pytest.mark.parametrize("tool,command", RAW_IMAGE_TOOLS, ids=[t[0] for t in RAW_IMAGE_TOOLS])
async def test_raw_tool_available(tool: str, command: str) -> None:
    """Tool is installed in RAW image and returns exit code 0."""
    config = SchedulerConfig(images_dir=images_dir)

    async with Scheduler(config) as scheduler:
        result = await scheduler.run(
            code=command,
            language=Language.RAW,
        )

        assert result.exit_code == 0, (
            f"Tool '{tool}' failed in raw image.\n"
            f"Command: {command}\n"
            f"Exit code: {result.exit_code}\n"
            f"Stdout: {result.stdout}\n"
            f"Stderr: {result.stderr}"
        )


# Python image tool tests - use Python subprocess to verify tools
PYTHON_IMAGE_TOOLS = [
    ("python", "import sys; print(sys.version)"),
    (
        "pip",
        "import subprocess; r = subprocess.run(['pip', '--version'], capture_output=True, text=True); print(r.stdout); exit(r.returncode)",
    ),
    (
        "curl",
        "import subprocess; r = subprocess.run(['curl', '--version'], capture_output=True, text=True); print(r.stdout); exit(r.returncode)",
    ),
    (
        "git",
        "import subprocess; r = subprocess.run(['git', '--version'], capture_output=True, text=True); print(r.stdout); exit(r.returncode)",
    ),
    (
        "gcc",
        "import subprocess; r = subprocess.run(['gcc', '--version'], capture_output=True, text=True); print(r.stdout); exit(r.returncode)",
    ),
]


@pytest.mark.parametrize("tool,code", PYTHON_IMAGE_TOOLS, ids=[t[0] for t in PYTHON_IMAGE_TOOLS])
async def test_python_tool_available(tool: str, code: str) -> None:
    """Tool is installed in Python image."""
    config = SchedulerConfig(images_dir=images_dir)

    async with Scheduler(config) as scheduler:
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, (
            f"Tool '{tool}' failed in python image.\n"
            f"Code: {code}\n"
            f"Exit code: {result.exit_code}\n"
            f"Stdout: {result.stdout}\n"
            f"Stderr: {result.stderr}"
        )


# JavaScript image tool tests - use Bun to verify tools
JAVASCRIPT_IMAGE_TOOLS = [
    ("bun", "console.log(Bun.version)"),
    ("curl", "const p = Bun.spawn(['curl', '--version']); await p.exited; process.exit(p.exitCode)"),
    ("git", "const p = Bun.spawn(['git', '--version']); await p.exited; process.exit(p.exitCode)"),
]


@pytest.mark.parametrize("tool,code", JAVASCRIPT_IMAGE_TOOLS, ids=[t[0] for t in JAVASCRIPT_IMAGE_TOOLS])
async def test_javascript_tool_available(tool: str, code: str) -> None:
    """Tool is installed in JavaScript image."""
    config = SchedulerConfig(images_dir=images_dir)

    async with Scheduler(config) as scheduler:
        result = await scheduler.run(
            code=code,
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0, (
            f"Tool '{tool}' failed in javascript image.\n"
            f"Code: {code}\n"
            f"Exit code: {result.exit_code}\n"
            f"Stdout: {result.stdout}\n"
            f"Stderr: {result.stderr}"
        )


# =============================================================================
# Functional tests - verify tools actually work, not just exist
# =============================================================================

# RAW functional tests (shell commands)
RAW_FUNCTIONAL_TESTS = [
    ("jq-parse", 'echo \'{"name": "test"}\' | jq -r .name', "test"),
    ("gzip-roundtrip", "echo 'hello' | gzip | gunzip", "hello"),
]


@pytest.mark.parametrize("name,command,expected", RAW_FUNCTIONAL_TESTS, ids=[t[0] for t in RAW_FUNCTIONAL_TESTS])
async def test_raw_functional(name: str, command: str, expected: str) -> None:
    """Shell command works correctly in RAW image."""
    config = SchedulerConfig(images_dir=images_dir)

    async with Scheduler(config) as scheduler:
        result = await scheduler.run(
            code=command,
            language=Language.RAW,
        )

        assert result.exit_code == 0, (
            f"Command failed: {name}\nCommand: {command}\nExit code: {result.exit_code}\nStderr: {result.stderr}"
        )
        assert expected in result.stdout, (
            f"Expected '{expected}' in output.\nCommand: {command}\nStdout: {result.stdout}"
        )


# Python functional tests (native Python code)
PYTHON_FUNCTIONAL_TESTS = [
    ("print", "print(2 + 2)", "4"),
    ("json", 'import json; print(json.loads(\'{"name": "test"}\')["name"])', "test"),
    (
        "pip-list",
        "import subprocess; r = subprocess.run(['pip', 'list', '--format=columns'], capture_output=True, text=True); print(r.stdout.splitlines()[0])",
        "Package",
    ),
]


@pytest.mark.parametrize("name,code,expected", PYTHON_FUNCTIONAL_TESTS, ids=[t[0] for t in PYTHON_FUNCTIONAL_TESTS])
async def test_python_functional(name: str, code: str, expected: str) -> None:
    """Python code works correctly in Python image."""
    config = SchedulerConfig(images_dir=images_dir)

    async with Scheduler(config) as scheduler:
        result = await scheduler.run(
            code=code,
            language=Language.PYTHON,
        )

        assert result.exit_code == 0, (
            f"Code failed: {name}\nCode: {code}\nExit code: {result.exit_code}\nStderr: {result.stderr}"
        )
        assert expected in result.stdout, f"Expected '{expected}' in output.\nCode: {code}\nStdout: {result.stdout}"


# JavaScript functional tests (native Bun/JS code)
JAVASCRIPT_FUNCTIONAL_TESTS = [
    ("print", "console.log(2 + 2)", "4"),
    ("json", 'console.log(JSON.parse(\'{"name": "test"}\').name)', "test"),
]


@pytest.mark.parametrize(
    "name,code,expected", JAVASCRIPT_FUNCTIONAL_TESTS, ids=[t[0] for t in JAVASCRIPT_FUNCTIONAL_TESTS]
)
async def test_javascript_functional(name: str, code: str, expected: str) -> None:
    """JavaScript code works correctly in JavaScript image."""
    config = SchedulerConfig(images_dir=images_dir)

    async with Scheduler(config) as scheduler:
        result = await scheduler.run(
            code=code,
            language=Language.JAVASCRIPT,
        )

        assert result.exit_code == 0, (
            f"Code failed: {name}\nCode: {code}\nExit code: {result.exit_code}\nStderr: {result.stderr}"
        )
        assert expected in result.stdout, f"Expected '{expected}' in output.\nCode: {code}\nStdout: {result.stdout}"

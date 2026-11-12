"""The import gate of exec_sandbox/__init__.py: an unsupported OS gets ConfigError PLATFORM_UNSUPPORTED."""

import platform
import subprocess
import sys
import textwrap
from pathlib import Path

_GATE = textwrap.dedent(
    """
    import psutil

    psutil.LINUX = psutil.MACOS = False  # a host that is neither Linux nor macOS
    try:
        from exec_sandbox import Scheduler
    except Exception as e:
        if getattr(e, "code", None) != "PLATFORM_UNSUPPORTED":
            raise
        print(type(e).__name__, e.code, e.action, e.context["platform"])
    else:
        print("imported")
    """
)
"""The pattern that exec_sandbox/__init__.py documents, in a fresh interpreter."""


def test_unsupported_platform_is_refused_by_code(tmp_path: Path) -> None:
    """The gate raises before any other exec_sandbox module runs; the documented
    pattern catches it by its code."""
    run = subprocess.run(
        [sys.executable, "-c", _GATE], cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["ConfigError", "PLATFORM_UNSUPPORTED", "FIX_CONFIG", platform.system()]

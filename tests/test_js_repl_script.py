"""The guest's JavaScript REPL script run on this host under Bun (it needs Linux glibc):
start-up cases that no VM test can set up. Requires Linux and bun on PATH, no VM: the CI
runners have no Bun, so it runs on a Linux host where Bun is installed (the EC2 test host)."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import skip_unless_linux

_REPL = Path(__file__).parent.parent / "guest-agent" / "src" / "repl" / "scripts" / "js_repl.mjs"
_BUN = shutil.which("bun")

# Runs `bun <script>` under a seccomp filter: prctl(PR_SET_DUMPABLE, ...) fails with EPERM, every
# other call runs. The script takes the call from dlopen("libc.so.6"): no preload can replace it.
_NO_DUMPABLE = r"""
import ctypes, os, platform, sys
arch, nr = {"x86_64": (0xC000003E, 157), "aarch64": (0xC00000B7, 167)}[platform.machine()]
LD, JEQ, RET = 0x20, 0x15, 0x06
ALLOW, EPERM = 0x7FFF0000, 0x00050001
program = [
    (LD, 0, 0, 4), (JEQ, 1, 0, arch), (RET, 0, 0, ALLOW),  # another arch: no filter
    (LD, 0, 0, 0), (JEQ, 0, 3, nr),  # not prctl: allowed
    (LD, 0, 0, 16), (JEQ, 0, 1, 4),  # option PR_SET_DUMPABLE
    (RET, 0, 0, EPERM), (RET, 0, 0, ALLOW),
]
class Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]
class Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(Filter))]
filters = (Filter * len(program))(*program)
fprog = Program(len(program), filters)
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
assert libc.prctl(38, 1, None, 0, 0) == 0  # PR_SET_NO_NEW_PRIVS: a filter needs it without root
assert libc.prctl(22, 2, ctypes.addressof(fprog), 0, 0) == 0  # SECCOMP_MODE_FILTER
os.execv(sys.argv[1], sys.argv[1:])
"""

skip_unless_bun = pytest.mark.skipif(_BUN is None, reason="This test runs the script under Bun (bun on PATH)")


def _run_repl(code: str, *, dumpable_fails: bool) -> subprocess.CompletedProcess[bytes]:
    """Send `code` to the REPL as exec s1, with prctl(PR_SET_DUMPABLE) failing or not."""
    assert _BUN is not None
    command = [_BUN, str(_REPL)]
    if dumpable_fails:
        command = [sys.executable, "-c", _NO_DUMPABLE, *command]
    return subprocess.run(
        command, input=f"s1 {len(code.encode())}\n{code}".encode(), capture_output=True, timeout=60, check=False
    )


@skip_unless_linux
@skip_unless_bun
def test_a_failed_prctl_stops_the_repl_before_any_exec() -> None:
    """prctl(PR_SET_DUMPABLE, 0) failing stops the REPL at start (fail closed, as the Python
    REPL): non-zero exit, the reason on stderr, no exec runs. Without the failure the same
    exec runs and ends."""
    code = "console.log('ran')"

    works = _run_repl(code, dumpable_fails=False)
    fails = _run_repl(code, dumpable_fails=True)

    assert "ran" in works.stdout.decode(), works.stderr.decode()
    assert works.stderr.decode().startswith("__REPL_READY__\n")  # before any exec, once
    assert re.findall(r"^__SENTINEL_\w+?_-?\d+__$", works.stderr.decode(), re.MULTILINE) == ["__SENTINEL_s1_0__"]
    assert fails.returncode != 0
    assert "prctl(PR_SET_DUMPABLE, 0) failed" in fails.stderr.decode()
    assert "ran" not in fails.stdout.decode()
    assert "__SENTINEL_" not in fails.stderr.decode()
    assert "__REPL_READY__" not in fails.stderr.decode()  # the agent reports a REPL that did not start

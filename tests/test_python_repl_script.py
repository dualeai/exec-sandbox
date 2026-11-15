"""The guest's Python REPL script run on this host (it needs Linux glibc): start-up cases that
no VM test can set up, and the reseed flag. Requires Linux, no VM."""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tests.conftest import skip_unless_linux

_REPL = Path(__file__).parent.parent / "guest-agent" / "src" / "repl" / "scripts" / "python_repl.py"


def _sentinels(stderr: str) -> list[str]:
    """The sentinel lines of the REPL's stderr, in order: one per exec that ended."""
    return re.findall(r"^__SENTINEL_\w+?_-?\d+__$", stderr, re.MULTILINE)


def _frame(sentinel_id: str, code: str, *, reseed: bool) -> str:
    """One exec as the agent writes it: the header (sentinel id, code length in bytes, reseed
    flag), then the code."""
    return f"{sentinel_id} {len(code.encode())} {int(reseed)}\n{code}"


def _run_execs(*execs: tuple[str, bool]) -> tuple[list[str], str]:
    """Send `execs` (code, reseed flag) to the REPL as the execs s0, s1, ... Returns (stdout
    lines, stderr)."""
    frames = "".join(_frame(f"s{i}", code, reseed=reseed) for i, (code, reseed) in enumerate(execs))
    run = subprocess.run(
        [sys.executable, str(_REPL)], input=frames.encode(), capture_output=True, timeout=60, check=False
    )
    return run.stdout.decode().splitlines(), run.stderr.decode()


def _run_repl(*codes: str) -> tuple[list[str], str]:
    """Send `codes` to the REPL as the execs s0, s1, ..., none with the reseed flag. Returns
    (stdout lines, stderr)."""
    return _run_execs(*((code, False) for code in codes))


def _printed(stdout: list[str]) -> list[str]:
    """The lines that user code printed: stdout without the end markers."""
    return [line for line in stdout if not line.startswith("__STDOUT_END_")]


@skip_unless_linux
def test_multiprocessing_imported_before_the_repl_still_forks(tmp_path: Path) -> None:
    """A sitecustomize imports multiprocessing and sets another start method before the REPL
    runs: the REPL still sets fork (forkserver hangs in the VM), as a later import would."""
    (tmp_path / "sitecustomize.py").write_text("import multiprocessing\nmultiprocessing.set_start_method('spawn')\n")
    code = "import multiprocessing; print(multiprocessing.get_start_method())"

    run = subprocess.run(
        [sys.executable, str(_REPL)],
        input=_frame("s1", code, reseed=False).encode(),
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert run.stdout.decode().splitlines()[0] == "fork", run.stderr.decode()
    assert _sentinels(run.stderr.decode()) == ["__SENTINEL_s1_0__"]


@skip_unless_linux
def test_a_failed_prctl_stops_the_repl_before_any_exec(tmp_path: Path) -> None:
    """prctl(PR_SET_DUMPABLE, 0) failing (a sitecustomize makes the call return -1) stops the
    REPL at start (fail closed): non-zero exit, the reason on stderr, no exec runs."""
    (tmp_path / "sitecustomize.py").write_text(
        "import _ctypes\n"
        "class _Fails(_ctypes.CFuncPtr):\n"
        "    _flags_ = _ctypes.FUNCFLAG_CDECL\n"
        "    def __call__(self, *args):\n"
        "        return -1\n"
        "_ctypes.CFuncPtr = _Fails\n"
    )
    code = "print('ran')"

    run = subprocess.run(
        [sys.executable, str(_REPL)],
        input=_frame("s1", code, reseed=False).encode(),
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert run.returncode != 0
    assert "prctl(PR_SET_DUMPABLE, 0) failed" in run.stderr.decode()
    assert "ran" not in run.stdout.decode()
    assert "__SENTINEL_" not in run.stderr.decode()
    assert "__REPL_READY__" not in run.stderr.decode()  # the agent reports a REPL that did not start


@skip_unless_linux
def test_the_ready_line_comes_before_any_exec() -> None:
    """The REPL writes its ready line to stderr once set up, before it reads an exec: the agent
    writes the first exec only after it (a REPL that fails before it did not start)."""
    stdout, stderr = _run_repl("print('ran')")

    assert stderr.startswith("__REPL_READY__\n"), stderr
    assert stderr.count("__REPL_READY__") == 1
    assert stdout[:1] == ["ran"]


@skip_unless_linux
def test_user_names_do_not_break_the_loop() -> None:
    """User code binds names that the REPL's loop uses (compile, exec, ns, sys, int): every
    later exec still runs and ends."""
    stdout, stderr = _run_repl(
        "from re import compile",
        "ns = [1, 2]; exec = print; int = str",
        "sys = None",
        "print('ok')",
    )

    assert "ok" in stdout, stderr
    assert _sentinels(stderr) == [f"__SENTINEL_s{i}_0__" for i in range(4)], stderr


@skip_unless_linux
def test_replaced_original_streams_do_not_hide_the_end_of_an_exec() -> None:
    """User code replaces sys.__stdout__ and sys.__stderr__, then sys.stderr with an object
    whose write raises and that has no flush, then fails: each exec still ends, and the error
    shows on the original stderr."""
    stdout, stderr = _run_repl(
        "import io, sys; sys.__stdout__ = io.StringIO(); sys.__stderr__ = io.StringIO()",
        "class Broken:\n    def write(self, s):\n        raise OSError('broken')\nsys.stderr = Broken()",
        "1 / 0",
        "print('ok')",
    )

    assert "ok" in stdout, stderr
    assert _sentinels(stderr) == [f"__SENTINEL_s{i}_{code}__" for i, code in enumerate((0, 0, 1, 0))], stderr
    assert "ZeroDivisionError" in stderr


@skip_unless_linux
def test_a_finder_without_find_spec_does_not_break_multiprocessing() -> None:
    """A meta-path finder with no find_spec (the import system skips it) ahead of the REPL's
    hook: multiprocessing still imports and forks. On the host the hook is in place: on
    a warm-pool or snapshot-restored VM the warm-up has already imported multiprocessing,
    and no hook runs."""
    code = (
        "import sys\n"
        "class OldFinder:\n"
        "    def find_module(self, name, path=None):\n"
        "        return None\n"
        "sys.meta_path.insert(0, OldFinder())\n"
        "import multiprocessing\n"
        "print(multiprocessing.get_start_method())\n"
    )
    stdout, stderr = _run_repl(code)

    assert stdout[:1] == ["fork"], stderr
    assert _sentinels(stderr) == ["__SENTINEL_s0_0__"], stderr


@skip_unless_linux
def test_rebinding_sys_or_changing_builtins_does_not_break_the_repl() -> None:
    """`sys = None`, and builtin names rebound (`next = url`, `type = 1`), in user code:
    multiprocessing still imports with fork; a change to the builtins module itself does not
    reach the loop."""
    stdout, stderr = _run_repl(
        "sys = None; next = 'https://api.example/page2'; type = 1",
        "import multiprocessing; print(multiprocessing.get_start_method())",
        "import builtins; builtins.int = str",
        "print('ok')",
    )

    assert stdout == ["__STDOUT_END_s0__", "fork", "__STDOUT_END_s1__", "__STDOUT_END_s2__", "ok", "__STDOUT_END_s3__"]
    assert _sentinels(stderr) == [f"__SENTINEL_s{i}_0__" for i in range(4)], stderr


@skip_unless_linux
def test_closed_or_odd_streams_do_not_hide_the_end_of_an_exec() -> None:
    """A user stream whose flush raises SystemExit, sys.stderr = None before an error, and
    the original stdout and stderr closed: each exec still ends with its marker and its
    sentinel, and the error goes to the original stderr."""
    stdout, stderr = _run_repl(
        "import sys\nclass Odd:\n    def write(self, s):\n        return len(s)\n"
        "    def flush(self):\n        raise SystemExit(3)\nsys.stdout = Odd()",
        "sys.stdout = sys.__stdout__; sys.stderr = None; 1 / 0",
        "sys.stdout.close(); sys.__stderr__.close()",
        "pass",
    )

    assert "ZeroDivisionError" in stderr
    assert _sentinels(stderr) == [f"__SENTINEL_s{i}_{code}__" for i, code in enumerate((0, 1, 0, 0))], stderr
    assert [line for line in stdout if line.startswith("__STDOUT_END_")] == [f"__STDOUT_END_s{i}__" for i in range(4)]


@skip_unless_linux
def test_user_annotations_are_not_strings() -> None:
    """User code compiles without this script's `from __future__ import annotations`."""
    stdout, stderr = _run_repl("def f(x: int):\n    pass\nprint(f.__annotations__['x'] is int)")

    assert stdout[:1] == ["True"], stderr


@skip_unless_linux
def test_an_error_shows_whatever_user_code_did_to_stderr_or_traceback() -> None:
    """The traceback module blocked, then sys.stderr closed: each error still shows on the
    original stderr, and the REPL runs on."""
    stdout, stderr = _run_repl(
        "import sys; sys.modules['traceback'] = None",
        "1 / 0",
        "del sys.modules['traceback']; sys.stderr.close()",
        "[][1]",
        "print('ok')",
    )

    assert "ok" in stdout, stderr
    assert "ZeroDivisionError" in stderr
    assert "IndexError" in stderr
    assert _sentinels(stderr) == [f"__SENTINEL_s{i}_{code}__" for i, code in enumerate((0, 1, 0, 1, 0))], stderr


@skip_unless_linux
def test_a_file_that_gets_fd_1_never_gets_a_marker(tmp_path: Path) -> None:
    """User code closes fd 1 (os.fdopen in a with block); a file it opens later gets fd 1:
    the markers still go to stdout, and the file holds only the user's data."""
    target = tmp_path / "user.txt"
    stdout, stderr = _run_repl(
        "import os\nwith os.fdopen(1, 'w') as f:\n    f.write('x')",
        f"f = open({str(target)!r}, 'w'); f.write('user-data'); f.close()",
        "pass",
    )

    assert target.read_text() == "user-data"
    assert [line for line in stdout if "__STDOUT_END_" in line] == [
        "x__STDOUT_END_s0__",
        "__STDOUT_END_s1__",
        "__STDOUT_END_s2__",
    ], stderr


@skip_unless_linux
def test_a_future_statement_holds_for_later_execs() -> None:
    """`from __future__ import annotations` in one exec holds for the later ones, as in the
    Python REPL; the script's own does not apply to user code."""
    stdout, stderr = _run_repl(
        "def g(x: int):\n    pass\nprint(g.__annotations__['x'] is int)",
        "from __future__ import annotations",
        "def f(x: int):\n    pass\nprint(repr(f.__annotations__['x']))",
    )

    assert [line for line in stdout if "__STDOUT_END_" not in line] == ["True", "'int'"], stderr


@skip_unless_linux
def test_files_opened_after_closerange_never_get_protocol_text(tmp_path: Path) -> None:
    """User code closes fds 3..63, then opens files (they get the lowest free fds): the marker
    and the sentinel never go into them, and each exec still ends."""
    paths = [tmp_path / f"f{i}.txt" for i in range(3)]
    stdout, stderr = _run_repl(
        f"import os\nos.closerange(3, 64)\nfiles = [open(p, 'w') for p in {[str(p) for p in paths]!r}]",
        "for f in files:\n    f.write('mine')\n    f.close()",
        "pass",
    )

    assert [p.read_text() for p in paths] == ["mine"] * 3
    assert _sentinels(stderr) == [f"__SENTINEL_s{i}_0__" for i in range(3)], stderr
    assert [line for line in stdout if "__STDOUT_END_" in line] == [f"__STDOUT_END_s{i}__" for i in range(3)]


@skip_unless_linux
def test_a_daemon_child_does_not_keep_the_pipes_open() -> None:
    """A double-forked child that points fds 0-2 at /dev/null and lives on: once the REPL
    ends, both pipes end at once and its exit code shows (the child holds no copy of the
    REPL's protocol fds)."""
    daemon = (
        "import os\nif os.fork() == 0:\n    os.setsid()\n    if os.fork() == 0:\n"
        "        fd = os.open('/dev/null', os.O_RDWR)\n        for t in (0, 1, 2):\n            os.dup2(fd, t)\n"
        "        import time\n        time.sleep(30)\n    os._exit(0)\nos.wait()"
    )
    frames = "".join(_frame(f"s{i}", c, reseed=False) for i, c in enumerate((daemon, "import os; os._exit(3)")))
    started = time.monotonic()
    run = subprocess.run(
        [sys.executable, str(_REPL)], input=frames.encode(), capture_output=True, timeout=20, check=False
    )

    assert run.returncode == 3, run.stderr.decode()
    assert time.monotonic() - started < 10


@skip_unless_linux
def test_a_fork_keeps_a_user_fd_that_took_a_copy_number(tmp_path: Path) -> None:
    """User code closes the REPL's protocol copies (fds 1000 and 1001), puts its own file on
    fd 1000, then forks: the child still writes to that file (the at-fork hook closes only a
    fd that is still the pipe)."""
    target = tmp_path / "user.txt"
    code = (
        f"import os\nos.closerange(1000, 1002)\nfd = os.open({str(target)!r}, os.O_WRONLY | os.O_CREAT)\n"
        "os.dup2(fd, 1000)\nif os.fork() == 0:\n    os.write(1000, b'from the child')\n    os._exit(0)\nos.wait()"
    )
    _stdout, stderr = _run_repl(code)

    assert target.read_text() == "from the child", stderr
    assert _sentinels(stderr) == ["__SENTINEL_s0_0__"], stderr


@skip_unless_linux
def test_the_reseed_flag_gives_the_module_random_a_new_seed() -> None:
    """random.seed(42), then an exec without the flag: the stream goes on from the seed. The
    same with the flag: the module-level generator has a new seed from the kernel. A Random()
    object keeps its own seed."""
    continues = "print(random.random() == random.Random(42).random(), own.random() == random.Random(7).random())"
    stdout, stderr = _run_execs(
        ("import random; random.seed(42); own = random.Random(7)", False),
        (continues, False),
        ("random.seed(42); own = random.Random(7)", False),
        (continues, True),
    )

    assert _printed(stdout) == ["True True", "False True"], stderr
    assert _sentinels(stderr) == [f"__SENTINEL_s{i}_0__" for i in range(4)], stderr


@skip_unless_linux
def test_two_reseeds_of_one_seed_give_different_streams() -> None:
    """Two REPLs in the same state (random.seed(42)) each get the flag: the new seeds come from
    the kernel, so their streams differ (a fixed seed would give the same in both)."""
    runs = [_run_execs(("import random; random.seed(42)", False), ("print(random.random())", True)) for _ in range(2)]

    printed = [_printed(stdout) for stdout, _ in runs]
    assert all(len(lines) == 1 for lines in printed), runs
    assert printed[0] != printed[1]


@skip_unless_linux
def test_the_reseed_imports_neither_ssl_nor_random() -> None:
    """The reseed acts only on the modules that user code imported: it imports none (neither
    was loaded before it, so the check would see an import)."""
    stdout, stderr = _run_execs(
        ("import sys; before = set(sys.modules)", False),
        ("print(sorted({'_ssl', 'random'} & before), sorted({'_ssl', 'random'} & (set(sys.modules) - before)))", True),
    )

    assert _printed(stdout) == ["[] []"], stderr
    assert _sentinels(stderr) == ["__SENTINEL_s0_0__", "__SENTINEL_s1_0__"], stderr


@skip_unless_linux
def test_the_reseed_adds_32_kernel_bytes_to_openssl() -> None:
    """With _ssl loaded (here a fake that records its calls), the flag adds 32 bytes with 32
    bytes of entropy (RAND_add) before the code runs; an exec without the flag adds none."""
    stdout, stderr = _run_execs(
        (
            "import sys, types\ncalls = []\nfake = types.ModuleType('_ssl')\n"
            "fake.RAND_add = lambda data, entropy: calls.append((type(data).__name__, len(data), entropy))\n"
            "sys.modules['_ssl'] = fake",
            False,
        ),
        ("print(calls)", False),
        ("print(calls)", True),
    )

    assert _printed(stdout) == ["[]", "[('bytes', 32, 32.0)]"], stderr


@skip_unless_linux
def test_a_failed_reseed_ends_the_repl_before_its_code_runs() -> None:
    """RAND_add raises: the REPL shows the error and exits with code 1 at the exec with the flag,
    before its code runs, and reads no later exec (the agent starts a new REPL for it)."""
    stdout, stderr = _run_execs(
        (
            "import sys, types\nfake = types.ModuleType('_ssl')\n"
            "def RAND_add(data, entropy):\n    raise OSError('no entropy')\n"
            "fake.RAND_add = RAND_add\nsys.modules['_ssl'] = fake",
            False,
        ),
        ("print('ran')", True),
        ("print('ok')", False),
    )

    assert _printed(stdout) == [], stderr
    assert "OSError: no entropy" in stderr
    assert _sentinels(stderr) == ["__SENTINEL_s0_0__"], stderr

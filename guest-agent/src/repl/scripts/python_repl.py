from __future__ import annotations

import ctypes as _repl_ctypes
import os as _repl_os
import sys
import traceback
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO, Any

# Security: set PR_SET_DUMPABLE=0 to prevent ptrace from other UID 1000 processes.
# Must be done here (after exec) because begin_new_exec() always resets dumpable
# to 1, regardless of credential state. Blocks CVE-2022-30594 style attacks.
_repl_ctypes.CDLL("libc.so.6", use_errno=True).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE=0

sys.argv = ["-c"]

# Use __main__.__dict__ as exec namespace so functions have correct __globals__.
# This lets pickle serialize exec()'d functions by qualified name, and cloudpickle
# recognize __main__.__dict__ for minimal globals extraction.
import __main__  # noqa: E402

ns = __main__.__dict__
ns["__builtins__"] = __builtins__

# Force fork start method — Python 3.14 defaults to forkserver, which hangs in the
# single-process VM environment. fork is safe here (single-threaded, Linux).
import multiprocessing  # noqa: E402

multiprocessing.set_start_method("fork")


# Lazy-load cloudpickle: only imported when multiprocessing actually spawns a process.
# Saves ~100-150ms on REPL startup for scripts that never use multiprocessing.
# cloudpickle is safe to defer (pure Python, no import-time side effects).
# See: PEP 810 for future native lazy imports (Python 3.15+).
def _patch_cloudpickle() -> None:
    import copyreg  # noqa: PLC0415
    import io  # noqa: PLC0415
    import multiprocessing.reduction as _reduction  # noqa: PLC0415

    import cloudpickle  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    class _CloudForkingPickler(cloudpickle.Pickler):  # pyright: ignore[reportMissingImports, reportUnknownMemberType, reportUntypedBaseClass]
        _extra_reducers: ClassVar[dict[type, Callable[..., Any]]] = {}
        _copyreg_dispatch_table = copyreg.dispatch_table

        def __init__(self, *args: Any, **kwds: Any) -> None:
            super().__init__(*args, **kwds)  # pyright: ignore[reportUnknownMemberType]
            self.dispatch_table = self._copyreg_dispatch_table.copy()
            self.dispatch_table.update(self._extra_reducers)

        @classmethod
        def register(cls, type: type, reduce: Callable[..., Any]) -> None:
            cls._extra_reducers[type] = reduce

        @classmethod
        def dumps(cls, obj: Any, protocol: int | None = None) -> memoryview:
            buf = io.BytesIO()
            cls(buf, protocol).dump(obj)  # pyright: ignore[reportUnknownMemberType]
            return buf.getbuffer()

        loads: staticmethod[..., Any] = staticmethod(cloudpickle.loads)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    def _cloud_dump(obj: Any, file: IO[bytes], protocol: int | None = None) -> None:
        _CloudForkingPickler(file, protocol).dump(obj)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    _reduction.ForkingPickler = _CloudForkingPickler  # type: ignore[assignment]
    _reduction.dump = _cloud_dump  # type: ignore[assignment]


# Intercept multiprocessing.Process.start() to trigger lazy patching.
# After first call, restores the original start() to avoid overhead.
_mp_orig_start = multiprocessing.Process.start
_mp_patched = False


def _lazy_mp_start(self: multiprocessing.Process) -> None:
    global _mp_patched  # noqa: PLW0603
    if not _mp_patched:
        _mp_patched = True
        _patch_cloudpickle()
        multiprocessing.Process.start = _mp_orig_start  # type: ignore[assignment]
    return _mp_orig_start(self)


multiprocessing.Process.start = _lazy_mp_start  # type: ignore[assignment]

# PID guard: forked children inherit the REPL wrapper. Record parent PID so
# children that escape user code (via sys.exit(), exception, or fall-through)
# flush their output and terminate without writing a premature sentinel.
_repl_pid = _repl_os.getpid()

# Redirect sys.stdin to /dev/null so user code that reads stdin (input(),
# sys.stdin.read(), for line in sys.stdin, etc.) gets immediate EOF instead
# of blocking on the protocol pipe. The REPL loop reads from _stdin_buf
# (the original stdin buffer) for the length-prefixed command protocol.
_stdin_buf = sys.stdin.buffer
sys.stdin = open(_repl_os.devnull)  # noqa: SIM115, PTH123

while True:
    header = _stdin_buf.readline()
    if not header:
        break
    sentinel_id, code_len = header.decode().strip().split(" ", 1)
    code = _stdin_buf.read(int(code_len)).decode()
    exit_code = 0
    try:
        compiled = compile(code, "<exec>", "exec")
        exec(compiled, ns)  # noqa: S102
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        exit_code = 1
    # PID guard: if we're a forked child, flush output and terminate immediately
    # without writing a sentinel. Only the original REPL parent writes sentinels.
    if _repl_os.getpid() != _repl_pid:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001, S110
            pass
        _repl_os._exit(exit_code)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stderr.write(f"__SENTINEL_{sentinel_id}_{exit_code}__\n")
    sys.stderr.flush()

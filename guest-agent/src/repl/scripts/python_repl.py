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

    # Also patch the separate name binding in multiprocessing.connection:
    # Connection.send() uses _ForkingPickler captured at module import time,
    # which is a different reference than reduction.ForkingPickler.
    import multiprocessing.connection as _mp_conn  # noqa: PLC0415

    _mp_conn._ForkingPickler = _CloudForkingPickler  # type: ignore[attr-defined]  # noqa: SLF001


# Lazy cloudpickle patching: triggered on first Process.start() OR Pool().
# Python 3.14 changed default start method from fork to forkserver — Pool()
# workers are created by a server process, bypassing Process.start() in the
# parent. We intercept both entry points so whichever runs first applies the
# patch without adding import overhead to boot.
_mp_orig_start = multiprocessing.Process.start
_mp_patched = False


def _ensure_cloudpickle() -> None:
    global _mp_patched  # noqa: PLW0603
    if _mp_patched:
        return
    _mp_patched = True
    import contextlib  # noqa: PLC0415

    with contextlib.suppress(ImportError):
        _patch_cloudpickle()
    # Restore original start (remove proxy overhead for subsequent calls)
    multiprocessing.Process.start = _mp_orig_start  # type: ignore[assignment]


def _lazy_mp_start(self: multiprocessing.Process) -> None:
    _ensure_cloudpickle()
    return _mp_orig_start(self)


multiprocessing.Process.start = _lazy_mp_start  # type: ignore[assignment]

# Defer multiprocessing.pool import + Pool.__init__ patch to first Pool() use.
# `import multiprocessing.pool` pulls in ~10 transitive modules (queues,
# synchronize, etc.) — 30-80ms warm cache, 1-3s cold cache. Since most user
# code never uses Pool, we defer until actually needed.
# multiprocessing.Pool is a re-export of multiprocessing.pool.Pool — when user
# code does `multiprocessing.Pool()` or `from multiprocessing import Pool`,
# Python auto-imports multiprocessing.pool, which triggers our patch.
_mp_pool_patched = False


def _ensure_pool_patched() -> None:
    global _mp_pool_patched  # noqa: PLW0603
    if _mp_pool_patched:
        return
    _mp_pool_patched = True
    import multiprocessing.pool  # noqa: PLC0415

    _mp_pool_orig_init = multiprocessing.pool.Pool.__init__

    def _lazy_pool_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _ensure_cloudpickle()
        # Restore original init (one-shot proxy)
        multiprocessing.pool.Pool.__init__ = _mp_pool_orig_init  # type: ignore[assignment]
        return _mp_pool_orig_init(self, *args, **kwargs)

    multiprocessing.pool.Pool.__init__ = _lazy_pool_init  # type: ignore[assignment]


# Replace multiprocessing.Pool in the module dict with a wrapper that triggers
# cloudpickle + pool patching BEFORE Pool.__init__ runs. CPython populates
# Pool into multiprocessing.__dict__ eagerly via globals().update() at import
# time (no __getattr__ — see PEP 562). The value is a bound method on the
# default context that lazily does `from multiprocessing.pool import Pool`.
# We swap it so `multiprocessing.Pool()` or `from multiprocessing import Pool`
# gets our wrapper, which ensures cloudpickle is ready before any pickling.
_mp_orig_pool_method = multiprocessing.Pool  # type: ignore[attr-defined]


def _lazy_mp_pool(*args: Any, **kwargs: Any) -> Any:
    _ensure_cloudpickle()
    _ensure_pool_patched()
    # Restore original (one-shot wrapper)
    multiprocessing.Pool = _mp_orig_pool_method  # type: ignore[attr-defined]
    return _mp_orig_pool_method(*args, **kwargs)


multiprocessing.Pool = _lazy_mp_pool  # type: ignore[attr-defined]

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

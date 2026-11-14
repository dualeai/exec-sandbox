from __future__ import annotations

import _ctypes as _repl_ctypes
import os as _repl_os
import sys

# Not from typing (about 1 ms at start): type checkers read any TYPE_CHECKING name as True.
TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.machinery import ModuleSpec
    from multiprocessing.process import BaseProcess
    from types import ModuleType
    from typing import IO, Any, ClassVar


# Security: set PR_SET_DUMPABLE=0 to prevent ptrace from other UID 1000 processes.
# Must be done here (after exec) because begin_new_exec() always resets dumpable
# to 1, regardless of credential state. Blocks CVE-2022-30594 style attacks.
# Through the built-in _ctypes core: the ctypes package costs ~47 ms at start under
# TCG (~4.6 ms on HVF), this call ~1 ms. A failure stops the REPL (fail closed).
class _ReplInt(_repl_ctypes._SimpleCData):  # noqa: SLF001  # pyright: ignore[reportMissingTypeArgument, reportPrivateUsage]
    _type_ = "i"


class _ReplLibcFunction(_repl_ctypes.CFuncPtr):
    _flags_ = _repl_ctypes.FUNCFLAG_CDECL | _repl_ctypes.FUNCFLAG_USE_ERRNO
    _restype_ = _ReplInt


class _ReplLibc:
    _handle = _repl_ctypes.dlopen("libc.so.6", _repl_os.RTLD_NOW)


# PR_SET_DUMPABLE=0; the typing stubs want a CDLL where any object with a _handle works
if _ReplLibcFunction(("prctl", _ReplLibc()))(4, 0, 0, 0, 0) != 0:  # pyright: ignore[reportCallIssue, reportArgumentType]
    raise OSError(_repl_ctypes.get_errno(), "prctl(PR_SET_DUMPABLE, 0) failed")
del _ReplInt, _ReplLibcFunction, _ReplLibc

sys.argv = ["-c"]

# Use __main__.__dict__ as exec namespace so functions have correct __globals__.
# This lets pickle serialize exec()'d functions by qualified name, and cloudpickle
# recognize __main__.__dict__ for minimal globals extraction.
import __main__  # noqa: E402

ns = __main__.__dict__
ns["__builtins__"] = __builtins__

import builtins as _repl_builtins  # noqa: E402
from types import FunctionType as _ReplFunctionType  # noqa: E402

# The REPL's own functions (the command loop and the multiprocessing helpers) run with
# these globals, not with __main__.__dict__ (the user's namespace): user code may rebind
# any name there (`next = url`, `type = 1`, `sys = None`, `ns = []`), and none of their
# names come from it, builtins included: a copy, as user code may change the builtins
# module.
_repl_globals: dict[str, Any] = {"__builtins__": dict(vars(_repl_builtins)), "__name__": "__main__", "sys": sys}


def _repl_private(function: Callable[..., Any]) -> Any:
    """`function` with _repl_globals as its globals, bound there by its name."""
    private = _ReplFunctionType(
        function.__code__, _repl_globals, function.__name__, function.__defaults__, function.__closure__
    )
    _repl_globals[function.__name__] = private
    return private


# Lazy-load cloudpickle: only imported when multiprocessing actually spawns a process.
# Saves ~100-150ms on REPL startup for scripts that never use multiprocessing.
# cloudpickle is safe to defer (pure Python, no import-time side effects).
# See: PEP 810 for future native lazy imports (Python 3.15+).
@_repl_private
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
    # multiprocessing.queues binds its own name at import too: loaded before this patch (by
    # multiprocessing.pool or concurrent.futures.process), its queues would pickle with the
    # plain pickler; loaded after, it binds the patched one
    _mp_queues = sys.modules.get("multiprocessing.queues")
    if _mp_queues is not None:
        _mp_queues._ForkingPickler = _CloudForkingPickler  # type: ignore[attr-defined]  # noqa: SLF001


_mp_patched = False  # for type checkers: _ensure_cloudpickle's global is in _repl_globals
_repl_globals["_mp_patched"] = False


@_repl_private
def _ensure_cloudpickle() -> None:
    global _mp_patched  # noqa: PLW0603
    if _mp_patched:
        return
    _mp_patched = True
    import contextlib  # noqa: PLC0415

    with contextlib.suppress(ImportError):
        _patch_cloudpickle()


@_repl_private
def _setup_multiprocessing(multiprocessing: ModuleType) -> None:
    # Force fork start method — Python 3.14 defaults to forkserver, which hangs in the
    # single-process VM environment. fork is safe here (single-threaded, Linux).
    multiprocessing.set_start_method("fork", force=True)  # force: a sitecustomize may have set one

    # Lazy cloudpickle patching: on the first start of any process. Every process type
    # (multiprocessing.Process, the Pool and ProcessPoolExecutor workers of any context)
    # starts through BaseProcess.start, in this process, before a task is pickled.
    from multiprocessing.process import BaseProcess  # noqa: PLC0415

    mp_orig_start = BaseProcess.start

    def lazy_mp_start(self: BaseProcess) -> None:
        _ensure_cloudpickle()
        # Restore original start (remove proxy overhead for subsequent calls)
        BaseProcess.start = mp_orig_start  # type: ignore[method-assign]
        return mp_orig_start(self)

    BaseProcess.start = lazy_mp_start  # type: ignore[method-assign]


class _MultiprocessingImportHook:
    """Run _setup_multiprocessing when the first import of multiprocessing ends, before
    any code can use it. Imported at start, multiprocessing (pickle, socket) costs ~5 ms."""

    @staticmethod
    @_repl_private
    def find_spec(name: str, path: Any, target: Any = None) -> ModuleSpec | None:
        if name != "multiprocessing":
            return None
        # A finder with no find_spec is skipped, as the import system skips it
        specs = (
            find(name, path, target)
            for finder in sys.meta_path
            if finder is not _MultiprocessingImportHook and (find := getattr(finder, "find_spec", None)) is not None
        )
        spec: ModuleSpec | None = next((s for s in specs if s is not None), None)  # pyright: ignore[reportUnknownVariableType]
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader

        def exec_module(module: ModuleType) -> None:
            del loader.exec_module  # the loader's own method again
            loader.exec_module(module)
            # Only a run of the module ends the hook, not a find_spec alone
            if _MultiprocessingImportHook in sys.meta_path:
                sys.meta_path.remove(_MultiprocessingImportHook)
            _setup_multiprocessing(module)

        loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


_repl_globals["_MultiprocessingImportHook"] = _MultiprocessingImportHook

if "multiprocessing" in sys.modules:
    # Imported before this point (a sitecustomize or usercustomize): set it up now
    _setup_multiprocessing(sys.modules["multiprocessing"])
else:
    sys.meta_path.insert(0, _MultiprocessingImportHook)  # pyright: ignore[reportArgumentType]

# PID guard: forked children inherit the REPL wrapper. Record parent PID so
# children that escape user code (via sys.exit(), exception, or fall-through)
# flush their output and terminate without writing a premature stdout end
# marker or sentinel.
_repl_pid = _repl_os.getpid()

# Redirect sys.stdin to /dev/null so user code that reads stdin (input(),
# sys.stdin.read(), for line in sys.stdin, etc.) gets immediate EOF instead
# of blocking on the protocol pipe. The REPL loop reads from _stdin_buf
# (the original stdin buffer) for the length-prefixed command protocol.
_stdin_buf = sys.stdin.buffer
sys.stdin = open(_repl_os.devnull)  # noqa: SIM115, PTH123
# The stdout end marker and the sentinel go to copies of fds 1 and 2, made before user code
# runs: it may replace or close sys.stdout, sys.stderr and their originals, even fds 1 and 2
# (a later open() then gets fd 1, and the marker would go into that file). The
# copies are high fds (from 1000: os.closerange(3, 64) then open() does not get them), and
# each protocol write checks that its fd is still the pipe.
import fcntl as _repl_fcntl  # noqa: E402


def _protocol_copy(fd: int) -> int:
    try:
        return _repl_fcntl.fcntl(fd, _repl_fcntl.F_DUPFD_CLOEXEC, 1000)
    except OSError:  # a soft file limit of 1000 or less
        return _repl_os.dup(fd)


_stdout = sys.stdout
_stderr = sys.stderr
_stdout_fd = _protocol_copy(1)
_stderr_fd = _protocol_copy(2)
# The pipes, by identity (st_dev, st_ino): fd numbers can be reused, the pipes cannot
_stdout_pipe = (_repl_os.fstat(1).st_dev, _repl_os.fstat(1).st_ino)
_stderr_pipe = (_repl_os.fstat(2).st_dev, _repl_os.fstat(2).st_ino)


@_repl_private
def _close_protocol_copies() -> None:
    # A forked child never writes the protocol (the PID guard): without the copies, the
    # pipes end when the REPL and the processes that use fds 1 and 2 end. Only a fd
    # that is still the pipe: user code may have closed a copy and reused its number
    for fd, pipe in ((_stdout_fd, _stdout_pipe), (_stderr_fd, _stderr_pipe)):
        try:
            status = _repl_os.fstat(fd)
            if (status.st_dev, status.st_ino) == pipe:
                _repl_os.close(fd)
        except OSError:
            pass


_repl_os.register_at_fork(after_in_child=_close_protocol_copies)

import __future__ as _repl_future  # noqa: E402

# The flags of every `from __future__` feature (codeop's way to carry them between execs)
_future_flags = 0
for _repl_feature in _repl_future.all_feature_names:
    _future_flags |= getattr(_repl_future, _repl_feature).compiler_flag
_repl_globals.update(
    ns=ns,
    _repl_os=_repl_os,
    _repl_pid=_repl_pid,
    _stdin_buf=_stdin_buf,
    _stdout=_stdout,
    _stderr=_stderr,
    _stdout_fd=_stdout_fd,
    _stderr_fd=_stderr_fd,
    _stdout_pipe=_stdout_pipe,
    _stderr_pipe=_stderr_pipe,
    _future_flags=_future_flags,
)


@_repl_private
def _serve() -> None:  # noqa: PLR0915
    """The command loop (its globals: _repl_globals)."""

    def protocol_write(fds: tuple[int, int], pipe: tuple[int, int], text: str) -> None:
        # To the first of (copy, original) that is still the pipe; none: nothing.
        # Blocking first: user code may make fd 1 or 2 non-blocking (shared by the copy)
        for fd in fds:
            try:
                status = _repl_os.fstat(fd)
            except OSError:
                continue
            if (status.st_dev, status.st_ino) != pipe:
                continue
            _repl_os.set_blocking(fd, True)
            data = text.encode()
            while data:
                data = data[_repl_os.write(fd, data) :]
            return

    def flush_output() -> None:
        # User code may replace the streams with anything, or close them
        for fd in (1, 2):
            try:  # noqa: SIM105 - contextlib costs import time at start
                _repl_os.set_blocking(fd, True)
            except BaseException:  # noqa: BLE001, S110
                pass
        for stream in (sys.stdout, sys.stderr, _stdout, _stderr):
            try:  # noqa: SIM105
                stream.flush()  # pyright: ignore[reportOptionalMemberAccess] - None raises, caught
            except BaseException:  # noqa: BLE001, S110
                pass

    def show_error(error: BaseException) -> None:
        # traceback is imported at the first error, not at start (its imports cost ~11 ms).
        # The text cannot fail; it goes to the user's sys.stderr, else (None, closed, a
        # failing object) to the saved stderr fd
        try:
            text = "".join(__import__("traceback").format_exception(error))
        except BaseException:  # noqa: BLE001
            try:
                text = f"{type(error).__name__}: {error!r}\n"
            except BaseException:  # noqa: BLE001
                text = "Error: the raised exception cannot be shown\n"
        try:
            sys.stderr.write(text)  # pyright: ignore[reportOptionalMemberAccess] - None raises, caught
            sys.stderr.flush()  # pyright: ignore[reportOptionalMemberAccess]
        except BaseException:  # noqa: BLE001
            try:  # noqa: SIM105
                protocol_write((_stderr_fd, 2), _stderr_pipe, text)
            except BaseException:  # noqa: BLE001, S110
                pass

    # `from __future__` statements of an exec hold for the later ones, as in the Python REPL
    # (codeop); this script's own (annotations) does not (dont_inherit)
    future = 0
    # Ready: the agent writes the first exec only after it reads this line (a REPL that
    # fails before it did not start, and no user code ran)
    protocol_write((_stderr_fd, 2), _stderr_pipe, "__REPL_READY__\n")
    while True:
        header = _stdin_buf.readline()
        if not header:
            break
        sentinel_id, code_len = header.decode().strip().split(" ", 1)
        code = _stdin_buf.read(int(code_len)).decode()
        exit_code = 0
        try:
            compiled = compile(code, "<exec>", "exec", flags=future, dont_inherit=True)
            future |= compiled.co_flags & _future_flags
            exec(compiled, ns)  # noqa: S102
        except SystemExit as e:
            # Like a process exit status: sys.exit(-1) is 255, never the host's timeout code (-1).
            exit_code = e.code & 0xFF if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException as error:  # noqa: BLE001
            show_error(error)
            exit_code = 1
        flush_output()
        # PID guard: if we're a forked child, terminate now without writing the stdout end
        # marker or the sentinel. Only the original REPL parent writes them.
        if _repl_os.getpid() != _repl_pid:
            _repl_os._exit(exit_code)
        # End of this exec's stdout: the agent stops reading stdout here, without waiting for
        # the pipe to go idle. If it cannot be written, the agent's idle wait covers it.
        try:  # noqa: SIM105
            protocol_write((_stdout_fd, 1), _stdout_pipe, f"__STDOUT_END_{sentinel_id}__\n")
        except BaseException:  # noqa: BLE001, S110
            pass
        protocol_write((_stderr_fd, 2), _stderr_pipe, f"__SENTINEL_{sentinel_id}_{exit_code}__\n")


_serve()

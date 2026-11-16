"""Subprocess lifecycle utilities.

- start_managed_process: launch a subprocess with standard piping and process registration;
  with child_shell, a child that ends with this process, or that runs shell steps before
  its command, with no Python preexec_fn (Popen can then vfork)
- communicate_managed_process: run a short child to its end and collect its output
- exec_failed_error: the ConfigError of a launch whose exec failed (exit status 126 or 127)
- start_drain: the background task that drains stdout/stderr (prevents 64KB pipe deadlock)
- wait_for_socket: poll for a Unix socket created by a child process after fork+exec
"""

from __future__ import annotations

import asyncio
import ctypes
import errno
import logging
import os
import shutil
import signal
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, overload

from exec_sandbox.aio_utils import await_cancellation_safe, await_settled
from exec_sandbox.exceptions import ConfigError, ErrorCode
from exec_sandbox.platform_utils import HostOS, ProcessWrapper, child_env, detect_host_os
from exec_sandbox.process_registry import register_process, unregister_process
from exec_sandbox.resource_cleanup import cleanup_process
from exec_sandbox.system_probes import probe_setpriv_pdeathsig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

logger = logging.getLogger(__name__)


_SETPRIV: Final = ("setpriv", "--pdeathsig", "KILL", "--")
"""Sets the parent-death signal, then execs the rest (util-linux 2.33+, probe_setpriv_pdeathsig)."""

_PARENT_CHECK: Final = '[ "$PPID" = "$1" ] || exit 1'
"""After setpriv's prctl call: a parent that ended before the call sent no signal, and the
child has another parent now (child_shell step; "$1" is the pid of this process)."""


def child_shell(cmd: Sequence[str], *, steps: Sequence[tuple[str, str]]) -> list[str]:
    """`cmd` behind one /bin/sh that runs each step, then execs `cmd` in the same process.

    A step is a shell statement and the one value that it reads as "$1". Values and the
    arguments of `cmd` are positional parameters: no shell parses them. No step: `cmd`
    itself. The shell runs the exec of `cmd`, so Popen cannot report its errors: a `cmd[0]`
    that exec would not find, may not run or would refuse as no program raises here, as
    Popen would. Other exec errors end the shell with 126 or 127 (exec_failed_error).

    Raises:
        FileNotFoundError: `cmd[0]` is not on the PATH of child_env()
        PermissionError: `cmd[0]` is there, but not executable
        OSError: ENOEXEC: `cmd[0]` is not a program (see _PROGRAM_STARTS)
    """
    if not steps:
        return list(cmd)
    # Looked up as Popen does, on the PATH of child_env() (exec_errno)
    if (code := exec_errno(cmd[0], os.pathsep.join(os.get_exec_path(child_env())))) is not None:
        raise OSError(code, os.strerror(code), cmd[0])
    script = "".join(f"{statement}; shift; " for statement, _ in steps) + 'exec "$@"'
    return ["/bin/sh", "-c", script, "sh", *(value for _, value in steps), *cmd]


_PROGRAM_STARTS: Final = (
    b"#!",
    b"\x7fELF",
    b"\xca\xfe\xba\xbe",  # universal Mach-O
    b"\xfe\xed\xfa\xce",  # Mach-O, 32 and 64 bits, both byte orders
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
)
"""The first bytes of a file that the kernel execs: a script (#!), ELF (Linux), Mach-O and
universal Mach-O (macOS). Popen reports ENOEXEC for any other file that no binfmt_misc
entry claims (_binfmt_misc_claims), but a shell runs it as a shell script (POSIX): an
empty file then "runs" with exit 0."""

_BINFMT_MISC: Final = Path("/proc/sys/fs/binfmt_misc")
"""The binfmt_misc entries of this kernel (Linux): the kernel also execs a file one of them
claims, through its interpreter (a .pyc through Python, another CPU through qemu-user)."""

_BINPRM_BUF_SIZE: Final = 256
"""The first bytes of a file that the kernel reads to choose its format (binfmt_misc magic)."""


_runnable: set[tuple[str, str]] = set()
"""The (name, PATH) pairs that exec_errno found runnable: not looked up again (tests clear it)."""


def exec_errno(name: str, path: str) -> int | None:
    """The errno of an exec of `name` on `path` (os.pathsep-joined) that cannot start, or None:
    ENOENT when no file has the name, EACCES when no such file is executable, ENOEXEC when the
    file is not a program that the kernel runs. None is kept for the process (_runnable); an
    errno is not, so a binary installed after a failed launch starts at the next one (one
    changed later is seen at the exec). A few stat calls and a read of 256 bytes, as the exec
    itself makes; the binfmt_misc entries only for a file of no known start. A file this user
    may run but not read (mode 0711 of another user) is left to the exec."""
    if (name, path) in _runnable:
        return None
    found = shutil.which(name, path=path)
    if found is None:
        candidates = [Path(name)] if os.sep in name else [Path(folder) / name for folder in path.split(os.pathsep)]
        return errno.EACCES if any(candidate.is_file() for candidate in candidates) else errno.ENOENT
    try:
        with Path(found).open("rb") as file:
            start = file.read(_BINPRM_BUF_SIZE)
    except OSError:
        start = None
    if start is not None and not start.startswith(_PROGRAM_STARTS) and not _binfmt_misc_claims(found, start):
        return errno.ENOEXEC
    _runnable.add((name, path))
    return None


def _binfmt_misc_claims(path: str, start: bytes) -> bool:
    """Whether an enabled binfmt_misc entry claims the file at `path` that starts with
    `start`, as the kernel matches them (fs/binfmt_misc.c): its extension (after the last
    "." of the path), or its magic at its offset under its mask (the buffer the kernel
    reads, zero-filled after the end of the file). False when binfmt_misc is off, not
    mounted or unreadable (no Linux)."""
    try:
        if (_BINFMT_MISC / "status").read_text().strip() != "enabled":
            return False
        entries = [entry for entry in _BINFMT_MISC.iterdir() if entry.name not in ("status", "register")]
    except OSError:
        return False
    buffer = start.ljust(_BINPRM_BUF_SIZE, b"\0")
    raw_path = os.fsencode(path)
    extension = raw_path[raw_path.rfind(b".") + 1 :] if b"." in raw_path else None
    for entry in entries:
        try:  # bytes: an interpreter path or an extension need not be UTF-8
            status, *lines = entry.read_bytes().splitlines()
        except (OSError, ValueError):
            continue
        fields = {key: value for key, _, value in (line.partition(b" ") for line in lines)}
        if status != b"enabled":
            continue
        if b"extension" in fields:  # written as ".<extension>"
            if extension is not None and fields[b"extension"] == b"." + extension:
                return True
            continue
        try:
            offset, magic = int(fields[b"offset"]), bytes.fromhex(fields[b"magic"].decode("ascii"))
            mask = bytes.fromhex(fields[b"mask"].decode("ascii")) if b"mask" in fields else b"\xff" * len(magic)
        except (KeyError, ValueError):
            continue
        window = buffer[offset : offset + len(magic)]
        if len(window) == len(magic) and all((b ^ m) & k == 0 for b, m, k in zip(window, magic, mask, strict=True)):
            return True
    return False


_EXEC_FAILED: Final = {
    126: "a program of the launch cannot run: no exec bit or another CPU architecture",
    127: "a program of the launch was not found, or its ELF interpreter is missing",
}
"""Exit status of a launch whose exec of its next program failed (POSIX sh, util-linux
unshare and setpriv), and why. QEMU, gvproxy-wrapper and qemu-storage-daemon never exit
with 126 or 127."""


def exec_failed_error(
    returncode: int | None, *, binary: str, detail: str | None, context: dict[str, Any], hint: str | None
) -> ConfigError | None:
    """The ConfigError (HOST_DEPENDENCY) of a launch that ended with 126 or 127, before or
    instead of `binary`: an exec in it failed, as spawn_error gives for an exec that Popen
    runs (the shell of child_shell runs the exec of its command, so Popen cannot report
    it). It fails the same way on every retry. `detail` (the last line of its output) ends
    the message. None for any other end."""
    reason = None if returncode is None else _EXEC_FAILED.get(returncode)
    if reason is None:
        return None
    return ConfigError(
        f"Failed to start {binary}: exit status {returncode} ({reason})" + (f": {detail}" if detail else ""),
        code=ErrorCode.HOST_DEPENDENCY,
        context={**context, "binary": binary, "exit_code": returncode},
        hint=hint,
        instance_id=None,
    )


async def _launch_command(
    cmd: Sequence[str], *, steps: Sequence[tuple[str, str]], end_with_this_process: bool
) -> tuple[list[str], Callable[[], None] | None]:
    """The command and preexec_fn that start_managed_process spawns: `cmd` after `steps`
    (child_shell), and with `end_with_this_process` a child that the kernel SIGKILLs once the
    thread that spawns it ends: the event loop's thread, so with the loop on the main thread,
    this process (SIGKILL too).

    `setpriv --pdeathsig KILL` sets the signal where it works (probe_setpriv_pdeathsig;
    PR_SET_PDEATHSIG is kept across each exec, except of a setuid program such as sudo).
    Then the shell checks that its parent is still this process: a parent that ended before
    the prctl call sent no signal. No Python preexec_fn, so Popen can vfork: a fork copies
    the page tables of this process on the event loop's thread (4 ms at 100 MB of RSS, 28 ms
    at 2 GB; a vfork takes 0.3 ms). Else die_with_spawning_thread, which makes Popen fork,
    where the OS has the signal (Linux before util-linux 2.33), or the steps only.

    Raises:
        OSError: `cmd[0]` would not start (child_shell)
    """
    if not end_with_this_process:
        return child_shell(cmd, steps=steps), None
    if await probe_setpriv_pdeathsig():
        return [*_SETPRIV, *child_shell(cmd, steps=[(_PARENT_CHECK, str(os.getpid())), *steps])], None
    return child_shell(cmd, steps=steps), die_with_spawning_thread()


_PR_SET_PDEATHSIG: Final = 1
"""prctl option (<linux/prctl.h>): the signal this process gets when its parent thread ends."""


def die_with_spawning_thread() -> Callable[[], None] | None:
    """On Linux, a preexec_fn that asks the kernel to SIGKILL the child once the thread that
    spawns it ends (PR_SET_PDEATHSIG; kept across an exec, except of a setuid program such as
    sudo). That thread runs the event loop: with the loop on the main thread, the child ends
    with this process, SIGKILL included. None elsewhere (macOS has no such signal). The
    fallback of _launch_command on a host without setpriv: a preexec_fn makes Popen
    fork this process, not vfork."""
    if detect_host_os() != HostOS.LINUX:
        return None
    prctl = ctypes.CDLL(None, use_errno=True).prctl  # looked up here: the child only calls it
    args = (ctypes.c_int(_PR_SET_PDEATHSIG), ctypes.c_ulong(signal.SIGKILL), *(ctypes.c_ulong(0),) * 3)
    parent = os.getpid()

    def set_parent_death_signal() -> None:
        prctl(*args)
        if os.getppid() != parent:  # the parent ended before the call: no signal will come
            os._exit(1)

    return set_parent_death_signal


async def start_managed_process(
    cmd: Sequence[str],
    *,
    steps: Sequence[tuple[str, str]],
    end_with_this_process: bool,
    pass_fds: tuple[int, ...],
    on_process_started: Callable[[ProcessWrapper], None],
) -> ProcessWrapper:
    """Launch a subprocess with standard piping and process registration.

    Common boilerplate for all subprocesses (QEMU, gvproxy, QSD, qemu-img):
    - Wraps asyncio subprocess with ProcessWrapper for PID-reuse safety
    - Pipes stdout/stderr (prevents 64 KiB deadlock when paired with start_drain)
    - Gives stdin /dev/null: the host's stdin never reaches a child (QEMU's console
      chardev would read it into the guest and set it non-blocking). Its EOF is harmless:
      virtconsole ignores chardev events, so the guest's console output still arrives
    - Registers in process_registry for emergency Ctrl+C kill (prevents orphans)
    - Creates new session (isolates signal delivery, enables killpg)
    - Builds the launch (_launch_command): the `steps` run in one /bin/sh before the exec of
      `cmd` (child_shell: a cgroup join), and with `end_with_this_process` the child ends with
      this process (setpriv --pdeathsig, or the preexec_fn fallback)

    Callers handle their own error wrapping (ConfigError HOST_DEPENDENCY, TransientError GVPROXY_FAILED, etc.)
    and the steps after the launch (the line of a cgroup join, a socket wait, etc.).

    Args:
        cmd: Command and arguments.
        steps: Shell statements and the value each reads as "$1", run before the exec of `cmd`
            (Cgroup.join_step, Cgroup.join_or_stop_step).
        end_with_this_process: The kernel ends the child with this process (Linux). Not for a
            child that execs a setuid program (sudo drops the signal): its cgroup ends it.
        pass_fds: File descriptors to inherit (socket activation).
        on_process_started: Called with the process as soon as it is spawned and
            registered, before any cancellation can reach the caller (the caller then owns
            its cleanup).

    Returns:
        ProcessWrapper instance with stdout/stderr pipes.

    Raises:
        OSError: Process spawn failed, or `cmd[0]` would not start (child_shell).
        FileNotFoundError: Binary not found.
    """
    launch, preexec_fn = await _launch_command(cmd, steps=steps, end_with_this_process=end_with_this_process)

    async def spawn_and_publish() -> ProcessWrapper:
        proc = ProcessWrapper(
            await asyncio.create_subprocess_exec(
                *launch,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                preexec_fn=preexec_fn,
                pass_fds=pass_fds,
                env=child_env(),
            )
        )
        # Registration and the owner callback are synchronous: no cancellation point can
        # separate child creation from publication.
        register_process(proc)
        on_process_started(proc)
        return proc

    # asyncio.create_subprocess_exec() can create the OS child just before its
    # awaiting caller is cancelled. Defer the cancellation until the spawn has either
    # published the child or failed: no unregistered, unowned process at the await boundary.
    return await await_cancellation_safe(spawn_and_publish())


async def communicate_managed_process(
    cmd: Sequence[str],
    *,
    process_name: str,
    context_id: str,
) -> tuple[int, bytes, bytes]:
    """Run a transient child with cancellation-safe kill/reap/unregistration.

    Returns:
        (exit code, stdout, stderr) of the child

    Raises:
        OSError: the child did not start (start_managed_process)
    """

    async def run_and_cleanup() -> tuple[int, bytes, bytes]:
        owned_process: ProcessWrapper | None = None

        def publish_process(proc: ProcessWrapper) -> None:
            nonlocal owned_process
            owned_process = proc

        try:
            proc = await start_managed_process(
                cmd, steps=(), end_with_this_process=False, pass_fds=(), on_process_started=publish_process
            )
            stdout, stderr = await proc.communicate()
            return await proc.wait(), stdout, stderr  # communicate() reaped it
        finally:
            if owned_process is not None:
                delay = 1.0
                while True:
                    try:
                        if await cleanup_process(
                            owned_process,
                            process_name,
                            context_id,
                            term_timeout=3.0,
                            kill_first=False,
                        ):
                            unregister_process(owned_process)
                            break
                    except asyncio.CancelledError:
                        # Cancellation may land after communicate() completed
                        # but while cleanup is in flight. Keep this owned task
                        # alive until death is confirmed; the outer waiter still
                        # propagates its own cancellation.
                        continue
                    try:
                        await asyncio.sleep(delay)
                    except asyncio.CancelledError:
                        continue
                    delay = min(delay * 2, 30.0)

    operation = asyncio.create_task(run_and_cleanup())
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        operation.cancel()
        await await_settled(operation)
        if not operation.cancelled():
            operation.exception()
        raise


def start_drain(
    process: ProcessWrapper,
    *,
    process_name: str,
    context_id: str,
    stdout_handler: Callable[[str], None],
    stderr_handler: Callable[[str], None],
) -> asyncio.Task[None]:
    """Start the background task that drains the process's stdout and stderr to EOF.

    The task becomes the process's drain_task: while it runs, cleanup
    (ProcessWrapper.wait_with_timeout) waits for the exit without reading the pipes.
    Its error is logged (log_task_exception). The caller owns the task.

    Args:
        process: ProcessWrapper instance with stdout/stderr pipes
        process_name: Process identifier for the task name (e.g., "QEMU", "gvproxy")
        context_id: Context identifier (e.g., vm_id) for the task name
        stdout_handler: Callback for each stdout line
        stderr_handler: Callback for each stderr line
    """
    task = asyncio.create_task(
        drain_subprocess_output(process, stdout_handler=stdout_handler, stderr_handler=stderr_handler),
        name=f"{process_name}-drain-{context_id}",
    )
    task.add_done_callback(log_task_exception)
    process.drain_task = task
    return task


async def _lines(stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """Each line of `stream` to EOF, newline included. A line longer than the reader's limit
    (64 KiB) is dropped whole, not an end: a drain that ended there would leave the pipe to
    fill and its writer blocked (QEMU's stderr is a blocking pipe: a thread that holds
    QEMU's big lock and writes a warning would block, and QMP and SIGTERM with it; its
    console drops bytes on a full pipe instead)."""
    skipping = False  # in a line over the limit, up to its newline
    while True:
        try:
            line = await stream.readuntil(b"\n")
        except asyncio.IncompleteReadError as e:  # EOF: the last line has no newline
            if e.partial and not skipping:
                yield e.partial
            return
        except asyncio.LimitOverrunError as e:  # e.consumed bytes of the line, still buffered
            await stream.readexactly(e.consumed)
            skipping = True
            continue
        if not skipping:
            yield line
        skipping = False


async def drain_subprocess_output(
    process: ProcessWrapper,
    *,
    stdout_handler: Callable[[str], None],
    stderr_handler: Callable[[str], None],
) -> None:
    """Drain subprocess stdout/stderr concurrently to prevent 64KB pipe deadlock.

    Critical: Without concurrent draining, subprocess can deadlock when:
    1. Process writes to both stdout and stderr
    2. One pipe fills (64KB buffer)
    3. Process blocks waiting for read
    4. Sequential reader stuck on other pipe

    One asyncio.TaskGroup task reads each pipe, so neither pipe fills while the
    other is read.

    Args:
        process: ProcessWrapper instance with stdout/stderr pipes
        stdout_handler: Callback for each stdout line
        stderr_handler: Callback for each stderr line
    """

    async def read(stream: asyncio.StreamReader, handler: Callable[[str], None]) -> None:
        """Read `stream` until EOF."""
        async for line in _lines(stream):
            try:
                decoded = line.decode().rstrip()
                if decoded:
                    handler(decoded)
            except (UnicodeDecodeError, ValueError):
                pass  # A line that is not UTF-8, or a ValueError of the handler: the line is skipped

    # Concurrent readers, one per pipe (prevent deadlock)
    async with asyncio.TaskGroup() as tg:
        for stream, handler in ((process.stdout, stdout_handler), (process.stderr, stderr_handler)):
            if stream:
                tg.create_task(read(stream, handler))


def log_task_exception(task: asyncio.Task[None]) -> None:
    """Log exceptions from background tasks.

    Callback for asyncio.Task.add_done_callback() that properly logs any
    unhandled exceptions from background tasks. Prevents silent failures.

    Usage:
        task = asyncio.create_task(some_coroutine())
        task.add_done_callback(log_task_exception)

    Args:
        task: The completed asyncio task to check for exceptions
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Background task failed",
            extra={"task_name": task.get_name()},
            exc_info=exc,
        )


_SOCKET_POLL_SECONDS: Final = 0.002
"""Pause between two checks of wait_for_socket."""


@overload
async def wait_for_socket(
    path: Path,
    *,
    timeout: float,
    abort_check: Callable[[], None],
    keep_connection: Literal[True],
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]: ...


@overload
async def wait_for_socket(
    path: Path,
    *,
    timeout: float,
    abort_check: Callable[[], None],
    keep_connection: Literal[False],
) -> None: ...


async def wait_for_socket(
    path: Path,
    *,
    timeout: float,
    abort_check: Callable[[], None],
    keep_connection: bool,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
    """Wait for a Unix socket to appear and accept connections.

    Used after fork+exec of QEMU or QSD to wait for the process to create its
    QMP control socket. Polls the filesystem because there is no async event
    to await (the file is created by an external process).

    A connect before the file exists gets FileNotFoundError, one between the file's
    creation and the end of listen() ConnectionRefusedError (seen under load): the
    probe retries both, and ConnectionResetError too.

    Args:
        path: Path to the socket file.
        timeout: Maximum seconds to wait before raising TimeoutError.
        abort_check: Called at each poll iteration. Should raise to abort the wait
            early (e.g. when the spawning process has died).
        keep_connection: If True, return the connected (reader, writer) streams
            instead of closing the probe connection. Eliminates the TOCTOU gap
            between probe-close and the caller's real connect for single-client
            chardev sockets (e.g. QMP).

    Returns:
        ``(reader, writer)`` when *keep_connection* is True, else ``None``.

    Raises:
        TimeoutError: Socket did not appear or accept connections within *timeout* seconds.
        ConfigError: HOST_DEPENDENCY: this process may not connect (EACCES, EPERM): the
            socket belongs to another user (QEMU as qemu-vm) and this process lacks its group.
    """
    async with asyncio.timeout(timeout):
        while True:
            abort_check()
            try:
                r, w = await asyncio.open_unix_connection(str(path))
                if keep_connection:
                    return r, w
                w.close()
                await w.wait_closed()
                return None
            except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError):
                await asyncio.sleep(_SOCKET_POLL_SECONDS)
            except PermissionError as e:  # no wait helps: the mode or the groups must change
                raise socket_access_error(path, e) from e


def socket_access_error(path: Path, error: PermissionError) -> ConfigError:
    """The ConfigError (HOST_DEPENDENCY) of a socket this process may not connect to."""
    context: dict[str, object] = {"socket": str(path), "errno": error.errno, "groups": sorted(os.getgroups())}
    try:
        st = path.stat()
        context |= {"owner_uid": st.st_uid, "owner_gid": st.st_gid, "mode": oct(stat.S_IMODE(st.st_mode))}
    except OSError:
        pass
    return ConfigError(
        f"This process may not connect to {path}: {error.strerror}. A socket QEMU makes as qemu-vm needs "
        "this process in its group (add the user to group qemu-vm and log in again, or run under `sg qemu-vm`)",
        code=ErrorCode.HOST_DEPENDENCY,
        context=context,
        hint=None,
        instance_id=None,
    )

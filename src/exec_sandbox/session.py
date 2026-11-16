"""Session: one VM that stays alive across exec() calls.

Variables, imports, functions and files stay from one call to the next: the
REPL process in the VM keeps them.

Example:
    ```python
    async with Scheduler() as scheduler:
        async with await scheduler.new(VmSpec(language="python")) as session:
            await session.exec("x = 42")
            result = await session.exec("print(x)")
            assert result.stdout.strip() == "42"
    ```

Lifecycle:
    - scheduler.new(spec) or scheduler.reuse(instance_id) returns a Session
      that owns a VM.
    - exec() runs code in that VM; the state stays.
    - close() destroys the VM; the context manager calls it.
    - The idle timeout closes the session after a time with no call.
    - These failures close the session: cancellation, an unknown outcome of a
      guest call (OutcomeUnknownError GUEST_OUTCOME_UNKNOWN), each
      exec() error except InvalidInputError, exit 137, a VM found dead, a VM that
      QEMU stopped because the host disk is full (TransientError LOCAL_DISK at the
      next call), and the save() failures listed in Session.save. A save that the
      store refuses or does not confirm leaves the session open. Details:
      Session.exec and Session.save.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import os
import stat
import warnings
from collections.abc import (  # noqa: TC003 - Used at runtime for _guard() and on_stdout/on_stderr
    AsyncGenerator,
    Callable,
)
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO, TYPE_CHECKING, Self, cast

from exec_sandbox._logging import get_logger
from exec_sandbox.aio_utils import run_to_end
from exec_sandbox.constants import MAX_CODE_SIZE, MAX_FILE_SIZE_BYTES, MAX_TIMEOUT_SECONDS, SIGKILL_EXIT_CODE
from exec_sandbox.exceptions import (
    ErrorCode,
    InstanceGoneError,
    InstanceNotSavedWarning,
    InternalError,
    InvalidInputError,
    OutcomeUnknownError,
    SandboxError,
    SessionClosedError,
    no_instance_store,
)
from exec_sandbox.instance.ids import check_instance_id
from exec_sandbox.models import ExecutionResult, ExposedPort, FileInfo, InstanceRef, StartupTiming, TimingBreakdown
from exec_sandbox.qemu_vm import local_path_error
from exec_sandbox.vm_types import VmState

if TYPE_CHECKING:
    from exec_sandbox.instance.manager import InstanceManager, InstanceState
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.vm_manager import VmManager
    from exec_sandbox.vm_spec import VmSpec

logger = get_logger(__name__)

# Codes of a failed migration: save() closes the session on them unless the VM runs again.
_MIGRATION_CODES = frozenset({ErrorCode.MIGRATION_FAILED, ErrorCode.MIGRATION_UNSUPPORTED})  # what a save raises


def _bad_argument(message: str) -> InvalidInputError:
    """The InvalidInputError (ARGUMENT_INVALID) of an argument that breaks a documented limit."""
    return InvalidInputError(message, code=ErrorCode.ARGUMENT_INVALID, context=None, hint=None, instance_id=None)


def _session_closed(message: str) -> SessionClosedError:
    """The SessionClosedError (SESSION_CLOSED) of a call on a closed session."""
    return SessionClosedError(message, code=ErrorCode.SESSION_CLOSED, context=None, hint=None, instance_id=None)


class Session:
    """One VM that stays alive across exec() calls.

    Variables, imports, functions and files stay from one call to the next,
    until close(). To keep the state after close(), call save() first.

    The calls of one session (exec, the file calls, save) run one at a time, in the order
    they reach the session (write_file first opens a local Path source). One exception: the
    calls made after a save() run once its VM runs again, while the save stores the state
    (save() still returns only once the store committed it).
    """

    def __init__(
        self,
        *,
        vm: QemuVM,
        vm_manager: VmManager,
        spec: VmSpec,
        startup_timing: StartupTiming,
        instances: InstanceManager | None,
        instance: InstanceState | None,
        new_instance_id: str | None,
        idle_timeout_seconds: int,
        default_timeout_seconds: int,
    ) -> None:
        self._vm = vm
        self._spec = spec
        self._startup_timing = startup_timing
        self._instances = instances
        self._instance = instance
        # Id the first save gives a new instance (Scheduler.new(instance_id=...)); None: a new random id
        self._new_instance_id = new_instance_id
        # exec() and write_file() calls that succeeded, and that count at the pause of the last
        # successful save: they differ when changes are not saved (InstanceNotSavedWarning at close)
        self._writes = 0
        self._saved_writes = 0
        # _writes at the pause of the save in flight, from the time its VM runs again (None: none):
        # close() then counts them as saved, since the save goes on and stores them (or gives the
        # store's error to its caller)
        self._saving_writes: int | None = None
        # Calls in flight (a guest call until it returns, a save until its end): the idle timer
        # runs only when none is
        self._busy = 0
        # Saves of this session, one at a time up to their commit. Taken inside the guard, after
        # _exec_lock: a save keeps its place in the queue of calls.
        self._save_lock = asyncio.Lock()
        # A save in flight owns the instance files: close() leaves them to it (see save()).
        self._saving = False
        self._vm_manager = vm_manager
        self._idle_timeout_seconds = idle_timeout_seconds
        self._default_timeout_seconds = default_timeout_seconds
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._exec_count = 0
        self._exec_lock = asyncio.Lock()
        # loop.call_later handle of the idle deadline (None: not armed)
        self._idle_timer_handle: asyncio.TimerHandle | None = None
        # Strong ref to the fire-and-forget close task the timer callback spawns.
        self._idle_close_task: asyncio.Task[None] | None = None
        self._reset_idle_timer()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        """True when the session is closed."""
        return self._closed

    @property
    def vm_id(self) -> str:
        """Id of the VM."""
        return self._vm.vm_id

    @property
    def spec(self) -> VmSpec:
        """The VmSpec the VM was made from."""
        return self._spec

    @property
    def startup_timing(self) -> StartupTiming:
        """How the VM started (warm pool, L1 restore, cold boot or a saved instance),
        measured once by new() or reuse()."""
        return self._startup_timing

    @property
    def instance(self) -> InstanceRef | None:
        """The saved instance this session continues: set by reuse() and by each
        successful save() (save(instance_id=...) moves it to the new instance); None for a
        session from new() that was never saved."""
        return _ref(self._instance)  # the header stays in memory after close removes the files

    @property
    def exec_count(self) -> int:
        """Number of exec() calls that returned a result, whatever the exit code."""
        return self._exec_count

    @property
    def exposed_ports(self) -> list[ExposedPort]:
        """Port mappings (internal→external) of this session's VM, with their host ports."""
        return self._vm.exposed_ports

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def _guard(self) -> AsyncGenerator[Callable[[], None]]:
        """Acquire exec lock with session lifecycle checks.

        Encapsulates the preamble shared by all public methods:
        1. Check closed (fast path, no lock)
        2. Acquire _exec_lock
        3. Re-check closed (may have closed while waiting)
        4. Retire fast if the VM process is already dead or being destroyed
        5. Suspend the idle timer for the duration of the operation

        Idle means "no operation in flight": the timer is cancelled while an
        operation runs and re-armed when the last one finishes, so a long exec can
        never be destroyed mid-flight by its own idle deadline.

        It yields a function that frees _exec_lock before the call ends (save() frees it
        once its VM runs again); a second call of it does nothing, so the guard never frees
        a lock that another call holds.

        On the way out it also owns VM retirement for cancellation and
        unknown-outcome failures: any dispatched guest operation may still be
        running after CancelledError or an OutcomeUnknownError, so the session
        is closed before either propagates. The store's answer to a save
        (INSTANCE_CONFLICT, INSTANCE_SAVE_OUTCOME_UNKNOWN) is not such an outcome:
        the save sent nothing to the guest, so the session stays open. A failed call
        that retired the VM closes the session too: QEMU stopped it because the host
        disk is full (TransientError LOCAL_DISK), or the REPL exited with code 137
        between two calls (TransientError GUEST_UNAVAILABLE).
        """
        self._check_open()
        await self._exec_lock.acquire()
        held = True

        def release() -> None:
            nonlocal held
            if held:
                held = False
                self._exec_lock.release()

        try:
            if self._closed:
                raise _session_closed("Session closed while waiting for lock")
            if self._vm.state in (VmState.DESTROYING, VmState.DESTROYED) or self._vm.process.returncode is not None:
                # QEMU died while the session was idle: the process-exit
                # watcher (armed by a cold boot's guest wait, else by the first
                # execute) publishes DESTROYING; a restored VM before its first
                # execute shows its death only through returncode. Fail fast instead
                # of letting every operation time out one transient error at a time.
                await self._close_for_failure()
                raise _session_closed("Session VM was retired")
            self._busy += 1
            self._cancel_idle_timer()
            try:
                yield release
            except asyncio.CancelledError:
                # Every dispatched guest operation can outlive its cancelled
                # host waiter.  Retire the VM before permitting reuse, for
                # file/list operations as well as code execution.
                await self._close_for_failure()
                raise
            except Exception as e:
                # A guest call whose outcome is unknown (GUEST_OUTCOME_UNKNOWN: dispatch may have
                # reached the guest), or a call that found the VM unusable and retired it (it
                # changed the state checked above): no later call can use this VM.
                if (
                    isinstance(e, OutcomeUnknownError) and e.code is ErrorCode.GUEST_OUTCOME_UNKNOWN
                ) or self._vm.state in (
                    VmState.DESTROYING,
                    VmState.DESTROYED,
                ):  # pyright: ignore[reportUnnecessaryContains]
                    await self._close_for_failure()
                raise
            finally:
                self._busy -= 1
                if not self._closed and not self._busy:
                    self._reset_idle_timer()
        finally:
            release()

    def _check_open(self) -> None:
        """Raise SessionClosedError on a closed session. Every call checks it first, before
        its arguments and before any local file: on a closed session, whatever the
        arguments, the caller must open a new session (RESTART)."""
        if self._closed:
            raise _session_closed("Session is closed")

    async def _close_for_failure(self) -> None:
        """Close the session without letting a close-time error mask the
        primary failure (or swallow the CancelledError) being propagated."""
        try:
            await self._close()
        except Exception:
            logger.exception(
                "Session close failed while handling a prior failure",
                extra={"vm_id": self._vm.vm_id},
            )

    async def _resolve_content(self, content: bytes | Path | IO[bytes]) -> tuple[IO[bytes], bool]:
        """Convert content to a readable binary stream with size validation.

        Called BEFORE _guard() so local disk I/O doesn't hold the exec lock.

        Returns:
            (stream, owned): The stream and whether the caller should close it.
            ``owned=True`` means we created the stream (bytes→BytesIO, Path→file);
            ``owned=False`` means the caller passed an IO[bytes] they own.

        TOCTOU note: for Path, the type and size checks run on the open file
        (fstat), but the file can still grow during the reads. The guest agent
        also enforces the max file size, providing defense in depth.
        """
        if isinstance(content, Path):
            # One worker-thread job: the caller's path can be on a network disk. A
            # cancel waits for the job, then closes the file it opened.
            opened: list[IO[bytes]] = []
            try:
                await run_to_end(lambda: opened.append(_open_source(content)))
            except asyncio.CancelledError:
                for stream in opened:
                    stream.close()
                raise
            return opened[0], True
        if isinstance(content, bytes):
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise _bad_argument(f"Content is {len(content)} bytes, exceeds {MAX_FILE_SIZE_BYTES}")
            return io.BytesIO(content), True
        # IO[bytes] — caller owns the stream, we do NOT close it.
        if content.closed:  # every read would raise a builtin ValueError mid-write
            raise _bad_argument("Stream content is closed")
        # A zero-byte read moves nothing: it raises on a stream not open for reading and gives
        # str from a text stream (also a wrapped one, as NamedTemporaryFile("w+")). Both are
        # refused here: a read that fails after the header is sent closes the session.
        try:
            probe = cast("bytes | str", content.read(0))  # IO[bytes] types read() as -> bytes
        except OSError as e:  # io.UnsupportedOperation
            raise _bad_argument("Stream content is not readable") from e
        if isinstance(probe, str):
            raise _bad_argument("Stream content is a text stream: open it in binary mode")
        if probe:  # a stream that ignores the read size: these bytes are already lost
            raise _bad_argument("Stream content returned bytes for a zero-byte read")
        # Validate size only if the stream is seekable.
        if content.seekable():
            pos = content.tell()
            content.seek(0, 2)
            size = content.tell()
            content.seek(pos)
            if size - pos > MAX_FILE_SIZE_BYTES:
                raise _bad_argument(f"Stream content is {size - pos} bytes, exceeds {MAX_FILE_SIZE_BYTES}")
        return content, False

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------

    async def exec(
        self,
        code: str,
        *,
        timeout_seconds: int | None = None,
        env_vars: dict[str, str] | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Run code in the session's VM.

        A non-zero exit code does not close the session, with two exceptions:
        exit code 137, and a QEMU that stopped while it sent the result. In both
        cases exec() returns the result, then closes the session. Each error
        below closes the session too, except SessionClosedError and
        InvalidInputError.

        Args:
            code: Source code to execute.
            timeout_seconds: Execution timeout. Default: scheduler's default_timeout_seconds.
            env_vars: Environment variables for this execution: at most 100; a name of 1 to
                256 bytes, a value up to 4096 bytes; no control characters except tab. The
                guest refuses names that change how programs load or start, in any case (for
                example PATH, LD_PRELOAD, LD_LIBRARY_PATH, PYTHONPATH, NODE_OPTIONS, BASH_ENV).
            on_stdout: Callback for stdout chunks (streaming).
            on_stderr: Callback for stderr chunks (streaming).

        Returns:
            ExecutionResult with stdout, stderr, exit_code, timing info.

        Raises:
            TransientError: GUEST_UNAVAILABLE: the VM communication failed or timed out
                before the code was sent, or the guest could not start the code: its REPL
                did not start, or it exited with code 137 after the last call, for example
                on a SIGKILL from a background process (the session closes). A timeout of
                your code does not raise: it returns exit_code -1 with the reason in stderr.
                LOCAL_DISK: QEMU stopped the VM because the host disk is full (the
                session closes; the writes since the last save are lost). A call in
                flight at that moment raises OutcomeUnknownError with the same reason.
            OutcomeUnknownError: GUEST_OUTCOME_UNKNOWN: the transport failed after the
                code was sent; the session is closed. Check the side effects of the
                code before you retry.
            SessionClosedError: The session is closed (checked first, whatever the arguments).
            InvalidInputError: The session stays open. ARGUMENT_INVALID (timeout
                out of range), CODE_INVALID (code empty, only spaces, too large, or
                with null bytes), ENV_VAR_INVALID (a refused name, control characters,
                size limits), OUTPUT_LIMIT (stdout or stderr went above the guest
                limit).
            InternalError: Protocol failure (the session closes).
        """
        self._check_open()
        if timeout_seconds is not None and (timeout_seconds < 1 or timeout_seconds > MAX_TIMEOUT_SECONDS):
            raise _bad_argument(f"timeout_seconds must be between 1 and {MAX_TIMEOUT_SECONDS}, got {timeout_seconds}")
        if len(code) > MAX_CODE_SIZE:
            raise InvalidInputError(
                f"Code too large: {len(code)} characters exceeds the {MAX_CODE_SIZE} character limit",
                code=ErrorCode.CODE_INVALID,
                context={"code_size": len(code), "max_code_size": MAX_CODE_SIZE},
                hint=None,
                instance_id=None,
            )
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds

        async with self._guard():
            execute_start = asyncio.get_running_loop().time()
            try:
                result = await self._vm.execute(
                    code=code,
                    timeout_seconds=timeout,
                    env_vars=env_vars,
                    on_stdout=on_stdout,
                    on_stderr=on_stderr,
                )
            except InvalidInputError:
                # Caller bugs (bad input, output above the guest limit), not VM
                # failures: the session stays alive (a REPL that exited after its output went
                # past the limit is replaced at the next exec).
                raise
            except Exception:
                # VM failure — auto-close session. CancelledError is handled by
                # _guard, which retires the VM for every dispatched operation.
                await self._close_for_failure()
                raise

            execute_end = asyncio.get_running_loop().time()
            execute_ms = round((execute_end - execute_start) * 1000)

            self._exec_count += 1
            self._writes += 1

            # The timing the session measured (a copy: no new validation of the output)
            execution_result = result.model_copy(
                update={"timing": TimingBreakdown(execute_ms=execute_ms, connect_ms=result.timing.connect_ms)}
            )
            if result.exit_code == SIGKILL_EXIT_CODE or self._vm.state in (VmState.DESTROYING, VmState.DESTROYED):
                logger.warning(
                    "Retiring session VM after terminal execution result",
                    extra={
                        "vm_id": self._vm.vm_id,
                        "exit_code": result.exit_code,
                        "vm_state": self._vm.state.value,
                    },
                )
                await self._close()
            return execution_result

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    async def write_file(self, path: str, content: bytes | Path | IO[bytes], *, make_executable: bool = False) -> None:
        """Write a file in the VM at the given path.

        Args:
            path: Path relative to /home/user (for example "input.csv", "data/model.pkl").
            content: File content as bytes, a local Path, or an IO[bytes] stream.
                Bytes are not copied; a Path or a stream is read in 128 KiB chunks.
                The encoded chunks that wait to be sent use at most about 13.5 MiB per
                write in progress, whatever the file size.
                An IO[bytes] stream stays yours: exec-sandbox does not close it.
            make_executable: Give the file mode 0o755 (else 0o644).

        Raises:
            TransientError: GUEST_UNAVAILABLE when the transport failed before the call was sent (you can retry).
                LOCAL_DISK when this host cannot open the Path source now (I/O error, out
                of file handles or memory), or a source read fails during the write: nothing
                was written, the session stays open; a seekable stream is set back to its
                position, a non-seekable stream must be passed again from its start.
                LOCAL_DISK also when QEMU stopped the VM because the host disk under its
                home disk is full: the session closes.
            OutcomeUnknownError: GUEST_OUTCOME_UNKNOWN when the transport is lost after
                the call was sent: the write may have been stored (the session closes); also
                when the host disk fills while the call is in progress.
            SessionClosedError: The session is closed (checked first: whatever the
                arguments, no local file is opened).
            InvalidInputError: ARGUMENT_INVALID when the content exceeds the max file
                size limit, a stream is closed, not open for reading or a text stream, or
                a Path source is not a regular file (a folder, a FIFO, a device) or cannot
                be opened for a reason the caller can fix (missing, no read access, a file
                in its path, a symlink loop, a NUL byte); PATH_INVALID or GUEST_IO on guest
                agent validation or write failure. The session stays open.
            InternalError: GUEST_PROTOCOL: the guest broke the protocol or sent a wrong answer.
        """
        self._check_open()
        # Resolve content to a stream BEFORE acquiring the lock —
        # local disk I/O should not block other session operations.
        stream, owned = await self._resolve_content(content)
        try:
            async with self._guard():
                await self._vm.write_file(path, stream, make_executable=make_executable)
                self._writes += 1
        finally:
            if owned:
                stream.close()

    async def read_file(self, path: str, *, destination: Path | IO[bytes]) -> None:
        """Read a file from the VM, and stream it to a local file or buffer.

        Memory use does not grow with the file size.

        Args:
            path: Path relative to /home/user (for example "output.csv").
            destination: Local file path or IO[bytes] buffer that gets the content.
                For a Path, content goes to `<destination>.<op_id>.tmp`, which
                replaces the file when the read is complete.
                For IO[bytes], content is written at the current position, and
                the stream stays yours: exec-sandbox does not close it.

        Raises:
            TransientError: GUEST_UNAVAILABLE on timeout or transport failure (you can
                retry on a live VM; a cancellation or an unknown outcome closes the
                session). LOCAL_DISK: this host could not write the local destination
                (disk full, quota, I/O error, out of file handles); or QEMU stopped the VM
                because the host disk under its home disk is full (the session closes).
            OutcomeUnknownError: GUEST_OUTCOME_UNKNOWN: the transport was lost after the
                call was sent, or the host disk filled while the call was in progress; the
                session closes.
            SessionClosedError: The session is closed (checked first, whatever the arguments).
            InvalidInputError: PATH_INVALID or GUEST_IO: file not found or validation failure.
                ARGUMENT_INVALID: the destination stream is closed, not writable or a text
                stream, or the caller can fix the destination Path (no access, a read-only
                disk, a file in its path, a folder at it, a NUL byte, no file name). The
                session stays open.
            InternalError: GUEST_PROTOCOL: the guest broke the protocol or sent a wrong answer.
        """
        self._check_open()
        if not isinstance(destination, Path):  # caller input: refused before the call takes the session
            if destination.closed:
                raise _bad_argument("read_file destination stream is closed")
            if not destination.writable():
                raise _bad_argument("read_file destination stream is not writable")
            try:  # writes nothing; a text stream (also a wrapped one, not an io.TextIOBase) takes only str
                destination.write(b"")
            except TypeError as e:
                raise _bad_argument("read_file destination stream is a text stream: open it in binary mode") from e
        async with self._guard():
            await self._vm.read_file(path, destination=destination)

    async def list_files(self, path: str = "") -> list[FileInfo]:
        """List the files of a folder in the VM.

        Args:
            path: Path relative to /home/user (empty string for /home/user itself).

        Returns:
            List of FileInfo entries with name, is_dir, and size.

        Raises:
            TransientError: GUEST_UNAVAILABLE: a timeout or a transport failure; you can
                retry on a live VM. LOCAL_DISK: QEMU stopped the VM because the host disk
                under its home disk is full; the session closes.
            OutcomeUnknownError: GUEST_OUTCOME_UNKNOWN: the transport was lost after the
                call was sent, or the host disk filled while the call was in progress; the
                session closes.
            SessionClosedError: The session is closed.
            InvalidInputError: PATH_INVALID or GUEST_IO: folder not found or validation failure.
            InternalError: GUEST_PROTOCOL: the guest broke the protocol or sent a wrong answer.
        """
        async with self._guard():
            return await self._vm.list_files(path)

    async def save(self, *, instance_id: str | None = None) -> InstanceRef:
        """Save the VM's state (memory, processes, files in /home/user) as a new generation.

        The first save of a session from new() makes the instance (generation 1)
        under the id given to new(instance_id=...), or a new random id; each later
        save, and each save of a session from reuse(), adds one generation. It
        succeeds only if nobody saved the instance since this session loaded or
        last saved it. The VM pauses while its state is written, then keeps
        running: the calls made after save() run then, while the save stores the state,
        and save() returns once the store committed it (a later save() waits for it).
        save() runs none of your code.

        With `instance_id` set to another id than this session's instance, the save
        makes generation 1 of a new instance with that id (an id that exists raises
        INSTANCE_CONFLICT: nothing is overwritten); once it is stored, the session
        continues the new instance (`session.instance`), and the previous instance
        stays as its last save left it. After INSTANCE_SAVE_OUTCOME_UNKNOWN of such a
        save, the session still continues the previous instance: call save() with the
        same `instance_id` again (a retry never stores a second copy). If the first save
        was stored, the retry raises INSTANCE_CONFLICT and the session does not move: the
        first error's error.context["commit_id"] is then in
        describe_instance(<instance_id>).recent_commits; reuse() that instance to continue
        from it. This session's own id gives a plain save.

        When close() runs during the save, the session is closed whatever the outcome: the
        "still open" in the list below holds only for a save that no close() met.

        A cancel while save() waits for an earlier exec() or file call leaves the session
        open; a cancel while it waits for an earlier save() to commit closes it. Once save()
        runs, a cancel closes the session, and the head write still finishes in the
        background: the save can be stored. A cancel gives no commit id: a later save is
        stored when describe_instance(id).generation is above session.instance.generation;
        a first save or a save under another id is stored when describe_instance(<that id>)
        finds the instance (list_instances() shows one under a random id).

        When the store refuses the save or does not confirm it (OutcomeUnknownError,
        InstanceGoneError), the VM runs on and the session stays open: close it, or
        save it again. After INSTANCE_CONFLICT of a plain save of a saved instance,
        another save holds the instance: a next plain save() fails at once (nothing is
        sent), and close() gives no InstanceNotSavedWarning. After INSTANCE_CONFLICT of a
        first save or of a save under another id, that id is taken: save under another id
        (unless it is the retry above). After INSTANCE_NOT_FOUND of a later save, the
        instance is gone: save under another id.

        Args:
            instance_id: None: this session's instance. Else 1 to 64 lowercase letters,
                digits, ".", "_" or "-", starting with a letter or digit (checked before
                any work).

        Raises:
            TransientError: INSTANCE_STORE_UNAVAILABLE or LOCAL_DISK (local disk full or
                files gone): nothing was saved; the session is still open.
                MIGRATION_FAILED: QEMU did not write the state (for example its disk is
                full); nothing was saved; the session stays open when the VM runs again
                (error.context["vm_resumed"]), else it closes.
            OutcomeUnknownError: INSTANCE_CONFLICT: another save was stored first, or the id
                given to new() is taken; nothing was saved. INSTANCE_SAVE_OUTCOME_UNKNOWN: the
                store did not confirm; a retry of the same save (the same instance_id) never
                stores a second copy (a plain save() after a failed save(instance_id=...) saves to
                this session's instance).
                The session is still open.
            SessionClosedError: The session is closed, or close() ran during this save
                and stopped it.
            InvalidInputError: SCHEDULER_STATE: the Scheduler is stopping (its exit refuses
                new saves); nothing was saved and the session is still open.
                INSTANCE_INVALID_ID: `instance_id` breaks the id rule; nothing was sent.
            ConfigError: INSTANCE_CONFIG: no instance store (instance_store +
                instance_keyring; checked first, so a closed session without a store
                raises this, not SessionClosedError), or the store refused this worker or
                has no conditional writes (S3 HTTP 501); the session is still open.
                HOST_DEPENDENCY: qemu-vm could not get access to the migration file, or this
                QEMU does not list the machine type the VM runs; nothing was saved; the
                session is still open.
                IMAGE_DAMAGED: a guest image does not match its digest; nothing was saved.
                MIGRATION_UNSUPPORTED: this QEMU cannot save the VM; the session closes
                unless the VM runs again (error.context["vm_resumed"]).
            InstanceGoneError: INSTANCE_NOT_FOUND: the instance was deleted meanwhile
                (also during a first save); nothing was saved; the session is still open.
        """
        if self._instances is None:
            raise no_instance_store(instance_id=None)
        self._check_open()
        if instance_id is not None:
            check_instance_id(instance_id)
        async with self._guard() as release, self._save_lock:
            if self._closed:  # close() ran while this save waited for an earlier one
                raise _session_closed("Session closed while waiting for an earlier save()")
            # A save under another id: the first save of that id; the session then continues it.
            # Read after both locks: an earlier save may have moved the session.
            own_id = self._instance.instance_id if self._instance is not None else self._new_instance_id
            moving = instance_id is not None and instance_id != own_id
            writes = self._writes  # no guest call runs until this save's pause
            resumed = False

            def on_resumed() -> None:  # the VM runs again: the calls queued after this save may run
                nonlocal resumed
                resumed = True
                self._saving_writes = writes
                release()

            self._saving = True
            try:
                state = await self._instances.save(
                    self._vm,
                    None if moving else self._instance,
                    new_instance_id=instance_id if moving else self._new_instance_id,  # for a first save only
                    resumed=on_resumed,
                )
            except (OutcomeUnknownError, InstanceGoneError) as e:
                # The store's answer: the VM runs on and the session stays open. After a close()
                # meanwhile, this error still goes to the caller, not SessionClosedError: it needs
                # error.context["commit_id"].
                await self._after_store_answer(e, instances=self._instances, moving=moving)
                raise
            except Exception as e:
                if self._closed and not resumed:  # close() destroyed the VM before it ran again
                    raise _session_closed("Session closed during save()") from e
                # By code, not by class: LOCAL_DISK and INSTANCE_STORE_UNAVAILABLE (also
                # TransientError) leave the VM running, so the session stays open.
                if isinstance(e, SandboxError) and e.code in _MIGRATION_CODES and not e.context.get("vm_resumed"):
                    await self._close_for_failure()  # the VM may stay paused: no later exec could run
                raise
            finally:
                self._saving = False
                self._saving_writes = None
                # close() left these files to the save. wait=False never suspends, so a cancel cannot
                # land between this release and the one of the new state below.
                if self._closed and self._instance is not None:
                    await self._instances.release(self._instance, wait=False)
            await self._release_after_save(state, instances=self._instances, moving=moving)
            self._instance = state
            self._saved_writes = writes
        return InstanceRef(instance_id=state.instance_id, generation=state.header.generation)

    async def _release_after_save(self, state: InstanceState, *, instances: InstanceManager, moving: bool) -> None:
        """The local files a successful save leaves. close() ran during the save: the new state's
        files go (the last step: the removal runs to its end in a worker thread even if a cancel
        comes then). Else the files no later save reads go
        in the background: the old instance's after a save under another id, and the new state's
        when another save is already on top of its commit."""
        if self._closed:
            await instances.release(state, wait=True)
            return
        if moving and self._instance is not None:  # the session moves to the new instance
            await instances.release(self._instance, wait=False)
        if state.etag is None:  # committed, then superseded: no later save reads these files
            await instances.release(state, wait=False)

    async def _after_store_answer(self, error: SandboxError, *, instances: InstanceManager, moving: bool) -> None:
        """What the session keeps of its instance after the store's answer to a save
        (INSTANCE_CONFLICT, INSTANCE_SAVE_OUTCOME_UNKNOWN, INSTANCE_NOT_FOUND). A save under
        another id (`moving`) leaves this session's instance as it was, whatever the answer."""
        if moving:
            return
        if error.code is ErrorCode.INSTANCE_CONFLICT and self._instance is not None and self._instance.etag is not None:
            # Another save of this instance won: a next plain save fails before it sends anything
            # (InstanceManager._save's check), and nothing reads these files again.
            dead = self._instance
            self._instance = dataclasses.replace(dead, etag=None)
            if not self._closed:  # when closed, save()'s `finally` releases them
                await instances.release(dead, wait=False)
        elif error.code is ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN and self._instance is None:
            # A first save that may have landed: a retry is create-only on the same id, so it
            # never makes a second instance (the error names the id the save used).
            self._new_instance_id = error.instance_id

    def _unsaved_instance(self) -> InstanceState | None:
        """The instance whose changes close() would lose: exec()/write_file() calls ran that no
        save stores (those after the last successful save, less those that a save in flight
        holds: its pause came after them, and its VM runs again), and a plain save could still
        store them (None: no such calls, or no such instance)."""
        saved = self._saved_writes if self._saving_writes is None else self._saving_writes
        instance = self._instance
        if self._writes == saved or instance is None or instance.etag is None:
            return None
        return instance

    async def close(self) -> None:
        """Close the session and destroy the VM. Never saves, never deletes the stored instance.

        Warns (InstanceNotSavedWarning) when the session continues a saved
        instance and exec()/write_file() ran since the last save, unless another save
        of that instance won (no plain save of this session can store them). A second call
        starts no new cleanup: it raises the error of the first one again, if any.
        Raises InternalError (CLEANUP_UNCONFIRMED) when the end of the QEMU process is not confirmed.
        """
        if not self._closed and (unsaved := self._unsaved_instance()) is not None:
            warnings.warn(
                f"Session closed with changes since the last save of instance {unsaved.instance_id} "
                f"(generation {unsaved.header.generation}); call save(), or save(instance_id=...) "
                "under another id, to keep them.",
                InstanceNotSavedWarning,
                stacklevel=2,
            )
        await self._close()

    async def _close(self) -> None:
        """Close the session and destroy the VM (no warning: failure and timeout paths).

        Idempotent: concurrent callers await the same cleanup task.

        Concurrency contract: close() does NOT acquire _exec_lock. Setting
        _closed=True immediately prevents new operations. An operation already
        inside _guard() may be interrupted by VM destruction; explicit close is
        the cancellation boundary, not a request-drain operation. Concurrent
        close callers await the same cleanup task. A cleanup failure is raised
        to every caller.
        """
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_impl())
        await asyncio.shield(self._close_task)

    async def _close_impl(self) -> None:
        """Perform the single shared cleanup operation."""
        self._cancel_idle_timer()
        # A save in flight now owns the instance's local files and releases them itself (save());
        # none starts after this point (_closed is set)
        release_instance = not self._saving

        logger.info(
            "Closing session",
            extra={
                "vm_id": self._vm.vm_id,
                "exec_count": self._exec_count,
            },
        )

        try:
            await self._vm_manager.destroy_vm(self._vm)
        finally:
            # The stored instance stays; its local files go
            if release_instance and self._instances is not None and self._instance is not None:
                await self._instances.release(self._instance, wait=True)
        # destroy_vm keeps failed workdir/cgroup cleanup owned for retry, and
        # publishes DESTROYED only after QEMU and gvproxy death is confirmed.
        # Only that process boundary may hide a terminal execution result.
        if self._vm.state is not VmState.DESTROYED:
            raise InternalError(
                f"VM process cleanup was not confirmed for session VM {self._vm.vm_id}",
                code=ErrorCode.CLEANUP_UNCONFIRMED,
                context={"vm_id": self._vm.vm_id, "operation": "session_close"},
                hint=None,
                instance_id=None,
            )

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """Exit async context manager - ensure cleanup.

        After a failure in the block, a close error is logged, not raised: it
        must not replace the error the caller is handling.
        """
        if exc_val is not None:
            await self._close_for_failure()
        else:
            await self.close()

    # -------------------------------------------------------------------------
    # Idle Timeout
    # -------------------------------------------------------------------------

    def _reset_idle_timer(self) -> None:
        """Arm (or re-arm) the idle deadline."""
        self._cancel_idle_timer()
        loop = asyncio.get_running_loop()
        self._idle_timer_handle = loop.call_later(self._idle_timeout_seconds, self._on_idle_timeout)

    def _cancel_idle_timer(self) -> None:
        """Cancel a pending idle deadline (no-op if it already fired)."""
        if self._idle_timer_handle is not None:
            self._idle_timer_handle.cancel()
            self._idle_timer_handle = None

    def _on_idle_timeout(self) -> None:
        """Idle deadline reached — close the session.

        Runs as a synchronous call_later callback, so it schedules the async
        close as a fire-and-forget task and holds a strong ref to it (the
        Session outlives it — close() publishes _close_task before any await).
        """
        self._idle_timer_handle = None
        if (unsaved := self._unsaved_instance()) is not None:
            logger.warning(
                "Idle timeout closes a session with changes since the last save",
                extra={"instance_id": unsaved.instance_id, "generation": unsaved.header.generation},
            )
        logger.info(
            "Session idle timeout",
            extra={
                "vm_id": self._vm.vm_id,
                "idle_timeout_seconds": self._idle_timeout_seconds,
                "exec_count": self._exec_count,
            },
        )
        self._idle_close_task = asyncio.create_task(self._close_for_failure())


def _open_source(path: Path) -> IO[bytes]:
    """write_file's source file, open for streaming (never loaded whole), after the
    type and size checks. Only a regular file: a FIFO or a device has no size to
    check and can block a read forever. O_NONBLOCK: the open of a FIFO with no
    writer returns at once (then the type check refuses it). Blocking:
    Session._resolve_content runs it in a worker thread. Its OSError or ValueError (a NUL
    byte): local_path_error."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise _bad_argument(f"Source {path.name} is not a regular file")
            if st.st_size > MAX_FILE_SIZE_BYTES:
                raise _bad_argument(f"File {path.name} is {st.st_size} bytes, exceeds {MAX_FILE_SIZE_BYTES}")
            os.set_blocking(fd, True)
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise
    except (OSError, ValueError) as e:
        raise local_path_error(e, failure="Source file cannot be opened", path=path, context={}) from e


def _ref(state: InstanceState | None) -> InstanceRef | None:
    """The public InstanceRef of `state` (None when there is no saved instance)."""
    return None if state is None else InstanceRef(instance_id=state.instance_id, generation=state.header.generation)

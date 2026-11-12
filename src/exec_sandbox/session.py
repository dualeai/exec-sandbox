"""Session - persistent VM for multi-step code execution.

A Session keeps a VM alive across multiple exec() calls, enabling
stateful workflows where variables, imports, and functions persist
between executions. The underlying REPL process maintains state.

Example:
    ```python
    async with Scheduler() as scheduler:
        async with await scheduler.new(VmSpec(language="python")) as session:
            await session.exec("x = 42")
            result = await session.exec("print(x)")
            assert result.stdout.strip() == "42"
    ```

Lifecycle:
    - Created via scheduler.new(spec) or scheduler.reuse(instance_id) (returns a
      Session that owns a VM)
    - exec() calls execute code in the same VM (state persists)
    - close() destroys the VM (called automatically by context manager)
    - Idle timeout auto-closes after inactivity (time between operations)
    - Failure closes: cancellation, unknown-outcome transport loss, exit 137,
      a VM found dead, and the save() failures listed in Session.save are all
      terminal — see Session.exec, Session.save and _guard()
"""

from __future__ import annotations

import asyncio
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

    The calls of one session (exec, the file calls, save) run one at a time.
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
        # Readable after close (the state's local files are removed then)
        self._instance_ref = _ref(instance)
        # exec()/write_file() succeeded since the last save (InstanceNotSavedWarning at a clean close)
        self._unsaved = False
        # A save in flight owns the instance files: close() leaves them to it (see save()).
        self._saving = False
        self._vm_manager = vm_manager
        self._idle_timeout_seconds = idle_timeout_seconds
        self._default_timeout_seconds = default_timeout_seconds
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._exec_count = 0
        self._exec_lock = asyncio.Lock()
        # loop.call_later handle for the idle deadline. cancel() is synchronous,
        # allocation-light, and a no-op once fired — no task, no CancelledError
        # to process, so no strong-ref set is needed.
        self._idle_timer_handle: asyncio.TimerHandle | None = None
        # Strong ref to the fire-and-forget close task the timer callback spawns.
        self._idle_close_task: asyncio.Task[None] | None = None
        self._reset_idle_timer()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        """Whether the session has been closed."""
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
        successful save(); None for a session from new() that was never saved."""
        return self._instance_ref

    @property
    def exec_count(self) -> int:
        """Number of successful exec() calls."""
        return self._exec_count

    @property
    def exposed_ports(self) -> list[ExposedPort]:
        """Resolved port mappings (internal→external) for this session's VM."""
        return self._vm.exposed_ports

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def _guard(self) -> AsyncIterator[None]:
        """Acquire exec lock with session lifecycle checks.

        Encapsulates the preamble shared by all public methods:
        1. Check closed (fast path, no lock)
        2. Acquire _exec_lock
        3. Re-check closed (may have closed while waiting)
        4. Retire fast if the VM process is already dead or being destroyed
        5. Suspend the idle timer for the duration of the operation

        Idle means "no operation in flight": the timer is cancelled while an
        operation runs and re-armed when it finishes, so a long exec can never
        be destroyed mid-flight by its own idle deadline.

        On the way out it also owns VM retirement for cancellation and
        unknown-outcome failures: any dispatched guest operation may still be
        running after CancelledError or OutcomeUnknownError, so the session
        is closed before either propagates.
        """
        self._check_open()
        async with self._exec_lock:
            if self._closed:
                raise _session_closed("Session closed while waiting for lock")
            if self._vm.state in (VmState.DESTROYING, VmState.DESTROYED) or self._vm.process.returncode is not None:
                # QEMU died while the session was idle: the process-exit
                # watcher (armed lazily on first execute) publishes DESTROYING;
                # before that first execute, death is visible only via
                # returncode. Fail fast instead of letting every operation
                # time out one transient error at a time.
                await self._close_for_failure()
                raise _session_closed("Session VM was retired")
            self._cancel_idle_timer()
            try:
                yield
            except asyncio.CancelledError:
                # Every dispatched guest operation can outlive its cancelled
                # host waiter.  Retire the VM before permitting reuse, for
                # file/list operations as well as code execution.
                await self._close_for_failure()
                raise
            except OutcomeUnknownError:
                # Dispatch may have reached the guest. Retire before exposing
                # the reconciliation-required error so no API can reuse a VM
                # whose command outcome is unknown.
                await self._close_for_failure()
                raise
            finally:
                if not self._closed:
                    self._reset_idle_timer()

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

        TOCTOU note: for Path, we stat() before open(). The file could grow
        between stat() and actual reads. The guest agent also enforces the
        max file size, providing defense in depth.
        """
        if isinstance(content, Path):
            if not content.exists():
                raise FileNotFoundError(f"Source file not found: {content}")
            file_size = content.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"File {content.name} is {file_size} bytes, exceeds {MAX_FILE_SIZE_BYTES}")
            # Return open file handle — streams from disk, never loads full content.
            return content.open("rb"), True
        if isinstance(content, bytes):
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"Content is {len(content)} bytes, exceeds {MAX_FILE_SIZE_BYTES}")
            return io.BytesIO(content), True
        # IO[bytes] — caller owns the stream, we do NOT close it.
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
        """Execute code in the session VM.

        State (variables, imports, functions) persists from previous exec() calls.
        Non-zero exit codes do not normally close the session. Two cases are
        terminal: exit 137 (reserved as a terminal result) and a VM whose QEMU
        died while delivering the result — both return the result, then close
        the session.

        Args:
            code: Source code to execute.
            timeout_seconds: Execution timeout. Default: scheduler's default_timeout_seconds.
            env_vars: Environment variables for this execution.
            on_stdout: Callback for stdout chunks (streaming).
            on_stderr: Callback for stderr chunks (streaming).

        Returns:
            ExecutionResult with stdout, stderr, exit_code, timing info.

        Raises:
            TransientError: GUEST_UNAVAILABLE: the VM communication failed or timed out
                before the code was sent (the session closes). A timeout of your code
                does not raise: it returns exit_code -1 with the reason in stderr.
            OutcomeUnknownError: GUEST_OUTCOME_UNKNOWN: the transport failed after the
                code was sent; the session is closed. Check the side effects of the
                code before you retry.
            SessionClosedError: The session is closed (checked first, whatever the arguments).
            InvalidInputError: The session stays open. ARGUMENT_INVALID (timeout
                out of range), CODE_INVALID (code empty, only spaces, too large, or
                with null bytes), ENV_VAR_INVALID (control characters, size
                limits), OUTPUT_LIMIT (stdout or stderr went above the guest
                limit).
            InternalError: Protocol failure (the session closes).
        """
        self._check_open()
        if timeout_seconds is not None and (timeout_seconds < 1 or timeout_seconds > MAX_TIMEOUT_SECONDS):
            raise _bad_argument(f"timeout_seconds must be between 1 and {MAX_TIMEOUT_SECONDS}, got {timeout_seconds}")
        if len(code) > MAX_CODE_SIZE:
            raise InvalidInputError(
                f"Code too large: {len(code)} bytes exceeds {MAX_CODE_SIZE} byte limit",
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
                # failures: the REPL is preserved and the session stays alive.
                raise
            except Exception:
                # VM failure — auto-close session. CancelledError is handled by
                # _guard, which retires the VM for every dispatched operation.
                await self._close_for_failure()
                raise

            execute_end = asyncio.get_running_loop().time()
            execute_ms = round((execute_end - execute_start) * 1000)

            self._exec_count += 1
            self._unsaved = True

            # Populate timing with session-measured values.
            execution_result = ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                execution_time_ms=result.execution_time_ms,
                external_cpu_time_ms=result.external_cpu_time_ms,
                external_memory_peak_mb=result.external_memory_peak_mb,
                external_cpu_nr_throttled=result.external_cpu_nr_throttled,
                timing=TimingBreakdown(execute_ms=execute_ms, connect_ms=result.timing.connect_ms),
                spawn_ms=result.spawn_ms,
                process_ms=result.process_ms,
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
                A Path streams from disk: it is never fully loaded in memory.
                An IO[bytes] stream stays yours: exec-sandbox does not close it.
            make_executable: Give the file mode 0o755 (else 0o644).

        Raises:
            TransientError: GUEST_UNAVAILABLE when the transport failed before the call was sent (you can retry).
                LOCAL_DISK when this host cannot open the Path source now (I/O error, out
                of file handles or memory), or a source read fails during the write: nothing
                was written, the session stays open; a seekable stream is set back to its
                position, a non-seekable stream must be passed again from its start.
            OutcomeUnknownError: GUEST_OUTCOME_UNKNOWN when the transport is lost after
                the call was sent: the write may have been stored (the session closes).
            SessionClosedError: The session is closed (checked first: whatever the
                arguments, no local file is opened).
            InvalidInputError: ARGUMENT_INVALID when the content exceeds the max file
                size limit, a stream is closed, not open for reading or a text stream, or
                a Path source is not a regular file (a folder, a FIFO, a device) or cannot
                be opened for a reason the caller can fix (missing, no read access, a file
                in its path, a symlink loop, a NUL byte); PATH_INVALID or GUEST_IO on guest
                agent validation or write failure. The session stays open.
        """
        self._check_open()
        # Resolve content to a stream BEFORE acquiring the lock —
        # local disk I/O should not block other session operations.
        stream, owned = await self._resolve_content(content)
        try:
            async with self._guard():
                await self._vm.write_file(path, stream, make_executable=make_executable)
                self._unsaved = True
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
                (disk full, quota, I/O error, out of file handles).
            SessionClosedError: The session is closed (checked first, whatever the arguments).
            InvalidInputError: PATH_INVALID or GUEST_IO: file not found or validation failure.
                ARGUMENT_INVALID: the destination stream is closed, not writable or a text
                stream, or the caller can fix the destination Path (no access, a read-only
                disk, a file in its path, a folder at it, a NUL byte, no file name). The
                session stays open.
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
                retry on a live VM.
            OutcomeUnknownError: GUEST_OUTCOME_UNKNOWN: the transport was lost after the
                call was sent; the session closes.
            SessionClosedError: The session is closed.
            InvalidInputError: PATH_INVALID or GUEST_IO: folder not found or validation failure.
        """
        async with self._guard():
            return await self._vm.list_files(path)

    async def save(self) -> InstanceRef:
        """Save the VM's state (memory, processes, files in /home/user) as a new generation.

        The first save of a session from new() makes the instance (generation 1)
        under the id given to new(instance_id=...), or a new random id; each later
        save, and each save of a session from reuse(), adds one generation. It
        succeeds only if nobody saved the instance since this session loaded or
        last saved it. The VM pauses while its state is written, then keeps
        running. save() runs none of your code.

        If the call is cancelled, the session closes and the head write still
        finishes in the background: the save can be stored (read the generation,
        as for VERIFY).

        Raises:
            TransientError: INSTANCE_STORE_UNAVAILABLE or LOCAL_DISK (local disk full or
                files gone): nothing was saved; the session is still open.
                MIGRATION_FAILED: QEMU did not write the state (for example its disk is
                full); nothing was saved; the session stays open when the VM runs again
                (error.context["vm_resumed"]), else it closes.
            OutcomeUnknownError: INSTANCE_CONFLICT: another save was stored first, or the id
                given to new() is taken; INSTANCE_SAVE_OUTCOME_UNKNOWN: the store did not confirm.
                The session is closed (VM discarded).
            SessionClosedError: The session is closed, or close() ran during this save
                and stopped it.
            ConfigError: INSTANCE_CONFIG: no instance store (instance_store +
                instance_keyring; checked first, so a closed session without a store
                raises this, not SessionClosedError), or the store refused this worker or
                has no conditional writes (S3 HTTP 501); the session is still open.
                HOST_DEPENDENCY: qemu-vm could not get access to the migration file; the
                session is still open.
                IMAGE_DAMAGED: a guest image does not match its digest; nothing was saved.
                MIGRATION_UNSUPPORTED: this QEMU cannot save the VM; the session closes
                unless the VM runs again (error.context["vm_resumed"]).
            InstanceGoneError: INSTANCE_NOT_FOUND: the instance was deleted meanwhile
                (also during a first save); the session is closed.
        """
        if self._instances is None:
            raise no_instance_store(instance_id=None)
        async with self._guard():
            self._saving = True
            try:
                state = await self._instances.save(
                    self._vm, self._instance, new_instance_id=self._new_instance_id if self._instance is None else None
                )
            except (OutcomeUnknownError, InstanceGoneError):
                # This VM diverged from the stored instance: discard it (no later save can win).
                await self._close_for_failure()
                raise
            except Exception as e:
                if self._closed:  # close() destroyed the VM during the save
                    raise _session_closed("Session closed during save()") from e
                # By code, not by class: LOCAL_DISK and INSTANCE_STORE_UNAVAILABLE (also
                # TransientError) leave the VM running, so the session stays open.
                if isinstance(e, SandboxError) and e.code in _MIGRATION_CODES and not e.context.get("vm_resumed"):
                    await self._close_for_failure()  # the VM may stay paused: no later exec could run
                raise
            finally:
                self._saving = False
                if self._closed and self._instance is not None:
                    await self._instances.release(self._instance)  # close() left these to the save
            if self._closed:  # close() ran during the save: these files are the save's to release
                await self._instances.release(state)
            self._instance = state
            self._instance_ref = _ref(self._instance)
            self._unsaved = False
        assert self._instance_ref is not None  # noqa: S101 - set just above
        return self._instance_ref

    async def close(self) -> None:
        """Close the session and destroy the VM. Never saves, never deletes the stored instance.

        Warns (InstanceNotSavedWarning) when the session continues a saved
        instance and exec()/write_file() ran since the last save. A second call
        does nothing. Raises InternalError (CLEANUP_UNCONFIRMED) when the end of the QEMU process is not confirmed.
        """
        if not self._closed and self._unsaved and self._instance_ref is not None:
            warnings.warn(
                f"Session closed with changes since the last save of instance {self._instance_ref.instance_id} "
                f"(generation {self._instance_ref.generation}); call save() to keep them.",
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
            # The stored instance stays; local files go (a save in flight releases them itself).
            if self._instances is not None and self._instance is not None and not self._saving:
                await self._instances.release(self._instance)
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
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """Exit async context manager - ensure cleanup."""
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
        if self._unsaved and self._instance_ref is not None:
            logger.warning(
                "Idle timeout closes a session with changes since the last save",
                extra={"instance_id": self._instance_ref.instance_id, "generation": self._instance_ref.generation},
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

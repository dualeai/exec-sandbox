"""Errors of exec-sandbox.

Every error the library raises at run time is a SandboxError (VmSpec, SchedulerConfig
and PortMapping raise pydantic's ValidationError for a bad value given in code). Its
class tells the caller what to do next:

    SandboxError (base, never raised itself)
    ├── TransientError        RETRY       repeat the call unchanged
    ├── OutcomeUnknownError   VERIFY      read the state before you do the work again
    ├── SessionClosedError    RESTART     open a new session
    ├── InvalidInputError     FIX_INPUT   correct the call
    ├── ConfigError           FIX_CONFIG  correct this worker's setup
    ├── RouteError            ROUTE       run the call on another worker
    ├── InstanceGoneError     RECREATE    start a new instance
    └── InternalError         REPORT      report a bug or a broken host

`error.code` (an ErrorCode) names the kind of failure, and each code belongs to exactly one
class: the constructor raises TypeError for a code of another class.
`error.action` (an ErrorAction) is the action of the class.

`str(error)` is one line, except when it quotes the output of a tool (the stderr of a
failed package install); the QEMU console of a crash or a boot timeout is in
`context["console_log"]`. The hint comes as a note (PEP 678), so every traceback
shows it. Never in a message, hint or context: key bytes, credentials, the
user's code or its output.
"""

from __future__ import annotations

import contextlib
import errno
import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping


class ErrorAction(StrEnum):
    """What the caller does next. Each error class has exactly one."""

    RETRY = "RETRY"
    """Repeat the same call unchanged, on this worker or another: it can succeed later. Bound the
    repeats (for example 3, with backoff): the library does not. The failed call did nothing you rely
    on (an exec did not send the code). If the failure closed the session, the work since its last
    save is lost (all of it, for a session never saved): open a new session, do that work again,
    then repeat the call."""

    VERIFY = "VERIFY"
    """Read the state before you do the work again: the side effects of the code; for a save,
    whether describe_instance(id).recent_commits holds error.context["commit_id"]. The state the
    call left is unknown (the code was sent, then the transport was lost; the store gave no clear
    answer to a save), or another save was stored first."""

    RESTART = "RESTART"
    """The session is closed: open a new one."""

    FIX_INPUT = "FIX_INPUT"
    """The call is wrong (an argument breaks a rule, or the scheduler cannot take the call now):
    correct it; the same call fails again. A session stays open."""

    FIX_CONFIG = "FIX_CONFIG"
    """This worker's setup is wrong (settings, keyring, store, binaries, images, Python extras,
    permissions or OS): change it; the same call fails again."""

    ROUTE = "ROUTE"
    """Repeat the call on another worker: this one has another CPU architecture, CPU vendor, guest
    CPU model or accelerator, lacks the saved machine type or a guest image the instance needs
    (and the store has no copy), runs a QEMU older than the one that saved the state, cannot
    read data that another exec-sandbox release wrote, or has a smaller budget than the VM
    needs (see the error's `required` field)."""

    RECREATE = "RECREATE"
    """The instance can no longer be used (deleted, expired, or its stored data is damaged): create a new one."""

    REPORT = "REPORT"
    """A bug in exec-sandbox or a broken host: report it with the log."""


class ErrorCode(StrEnum):
    """The kind of failure. Each code belongs to exactly one error class.

    The meaning of each code: the "Error codes" table of the README
    (https://github.com/dualeai/exec-sandbox#error-codes), or the docstring under
    each member below (an IDE shows it; help() does not).
    """

    # TransientError (RETRY)
    VM_BOOT_TIMEOUT = "VM_BOOT_TIMEOUT"
    """The guest agent did not answer in time while the VM booted."""
    QEMU_CRASHED = "QEMU_CRASHED"
    """QEMU exited or did not come up (boot, restore, snapshot build)."""
    OVERLAY_FAILED = "OVERLAY_FAILED"
    """qemu-img or qemu-storage-daemon could not make a disk image."""
    GVPROXY_FAILED = "GVPROXY_FAILED"
    """gvproxy did not start or did not become ready."""
    CAPACITY = "CAPACITY"
    """No resources became free within the admission timeout."""
    GUEST_UNAVAILABLE = "GUEST_UNAVAILABLE"
    """The guest could not take the call (transport failure, timeout or readiness failure before the
    call was sent), or could not start the code (the REPL did not start, or it exited with code 137
    between two calls). The code did not run."""
    GUEST_SILENT_AFTER_RESTORE = "GUEST_SILENT_AFTER_RESTORE"
    """The guest of a memory snapshot restore (an L1 memory snapshot or a saved instance) did not
    answer once its state loaded, or could not reseed its kernel random source then. A silent L1 memory
    snapshot is removed, and the VM boots cold."""
    LOCAL_DISK = "LOCAL_DISK"
    """A local file the call needs is missing (not a path the caller gave: that is ARGUMENT_INVALID);
    or this host cannot open or write a local file now (disk full, quota, I/O error, out of file
    handles or memory); or a local source cannot be read during a write_file (nothing was written;
    the session stays open); or the home disk of a new VM cannot fit on its filesystem; or the host
    disk under a VM's home disk is full (QEMU stopped the VM; the session closes)."""
    PACKAGE_DOWNLOAD = "PACKAGE_DOWNLOAD"
    """The package manager could not download (network error or timeout)."""
    MIGRATION_FAILED = "MIGRATION_FAILED"
    """QEMU did not save or load the VM state (QMP failure, timeout)."""
    ASSET_DOWNLOAD = "ASSET_DOWNLOAD"
    """A VM image download failed after its retries (network or checksum)."""
    INSTANCE_STORE_UNAVAILABLE = "INSTANCE_STORE_UNAVAILABLE"
    """The instance store failed, answered badly, or kept changing."""

    # OutcomeUnknownError (VERIFY)
    GUEST_OUTCOME_UNKNOWN = "GUEST_OUTCOME_UNKNOWN"
    """The call reached the guest, then the transport was lost. The code may have run."""
    INSTANCE_CONFLICT = "INSTANCE_CONFLICT"
    """Another save of the instance was stored first, or the id of a first save (Scheduler.new(instance_id=...),
    Session.save(instance_id=...)) is taken. This save was not stored."""
    INSTANCE_SAVE_OUTCOME_UNKNOWN = "INSTANCE_SAVE_OUTCOME_UNKNOWN"
    """The store did not confirm the save, and reading it back could not tell whether it was stored."""

    # SessionClosedError (RESTART)
    SESSION_CLOSED = "SESSION_CLOSED"
    """The session is closed."""

    # InvalidInputError (FIX_INPUT)
    CODE_INVALID = "CODE_INVALID"
    """The code is empty, only spaces, holds null bytes, or is too large."""
    ENV_VAR_INVALID = "ENV_VAR_INVALID"
    """An env var name or value breaks a rule."""
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    """stdout or stderr went above the guest limit. The session stays open."""
    PATH_INVALID = "PATH_INVALID"
    """A guest path breaks a rule (empty, too long, outside the sandbox, a folder)."""
    GUEST_IO = "GUEST_IO"
    """The guest could not read, list or write that path (not found, no access, guest disk full)."""
    ARGUMENT_INVALID = "ARGUMENT_INVALID"
    """An argument breaks a documented limit (timeout range, content size, a write_file source stream
    that is closed, not open for reading or a text stream, a read_file destination stream that is
    closed or not writable, a local source or destination path the caller can fix (missing, no access,
    a read-only disk, a file in the path, a folder at it, a symlink loop, too long, taken, a NUL byte,
    no file name, not a regular file), a package spec without a version, packages for a language that
    has none, a package list the guest refuses)."""
    PACKAGE_NOT_ALLOWED = "PACKAGE_NOT_ALLOWED"
    """A package is not in the allow-list."""
    PACKAGE_INSTALL_FAILED = "PACKAGE_INSTALL_FAILED"
    """The package manager refused the packages (no matching version, conflict, other failure)."""
    INSTANCE_INVALID_ID = "INSTANCE_INVALID_ID"
    """The value breaks the instance id rule."""
    SCHEDULER_STATE = "SCHEDULER_STATE"
    """The scheduler cannot take this call now (not started, already started, stopping)."""

    # ConfigError (FIX_CONFIG)
    HOST_DEPENDENCY = "HOST_DEPENDENCY"
    """A binary (QEMU, qemu-img, qemu-storage-daemon, gvproxy-wrapper), system user, cgroup or ACL
    support this worker needs is missing or too old, or this user cannot run the binary. A binary
    that hits a process, memory or file-handle limit raises TransientError instead."""
    EXTRA_MISSING = "EXTRA_MISSING"
    """A Python extra is not installed (the message names the pip command)."""
    SETTING_INVALID = "SETTING_INVALID"
    """A worker setting is not valid."""
    ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
    """A VM image (kernel, initramfs, base image) is not on this host and was not downloaded
    (auto download off, an images_dir that lacks it, no release to download it from)."""
    IMAGE_DAMAGED = "IMAGE_DAMAGED"
    """A local guest image does not match its digest: download or rebuild the images."""
    PLATFORM_UNSUPPORTED = "PLATFORM_UNSUPPORTED"
    """The OS is not Linux or macOS."""
    MIGRATION_UNSUPPORTED = "MIGRATION_UNSUPPORTED"
    """This worker's QEMU cannot save or load VM state: below QEMU 11.0 (at Scheduler start with an
    instance store, at a save or a restore), or a capability refused."""
    INSTANCE_CONFIG = "INSTANCE_CONFIG"
    """The worker's instance settings are wrong (store, keyring, a store that refuses or fails the probe)."""
    INSTANCE_KEY_MISSING = "INSTANCE_KEY_MISSING"
    """The key the head names is not in this keyring. The data is intact: do not delete the instance."""

    # RouteError (ROUTE)
    INSTANCE_HOST_MISMATCH = "INSTANCE_HOST_MISMATCH"
    """This worker's host values differ from the saved ones, or it cannot read the saved data."""
    CAPACITY_NEVER_FITS = "CAPACITY_NEVER_FITS"
    """The VM is larger than this worker's whole budget."""
    MIGRATION_REJECTED = "MIGRATION_REJECTED"
    """QEMU refused the saved state (incompatible or damaged stream). Callers see it only in the
    cause chain of the error they get: from reuse(), RouteError INSTANCE_HOST_MISMATCH on a QEMU
    older than the saver's, else InternalError INTERNAL. An L1 memory snapshot it refuses is
    removed, and the VM boots cold."""

    # InstanceGoneError (RECREATE)
    INSTANCE_NOT_FOUND = "INSTANCE_NOT_FOUND"
    """No instance with this id in the store (deleted, expired, or never saved)."""
    INSTANCE_EXPIRED = "INSTANCE_EXPIRED"
    """Part of the saved state is gone from the store; the head still names it."""
    INSTANCE_INTEGRITY = "INSTANCE_INTEGRITY"
    """Stored data failed its checks (changed, damaged, truncated)."""

    # InternalError (REPORT)
    CLEANUP_UNCONFIRMED = "CLEANUP_UNCONFIRMED"
    """The end of a QEMU or gvproxy process, or the release of VM resources, was not confirmed."""
    GUEST_PROTOCOL = "GUEST_PROTOCOL"
    """The guest sent a protocol or request error, or an answer of the wrong type."""
    INTERNAL = "INTERNAL"
    """A broken invariant (a bug)."""


class SandboxError(Exception):
    """Base of every exec-sandbox error. Never raised itself: catch it to catch them all.

    Attributes:
        message: One line that says what failed, except when it quotes the output of a tool
            (the stderr of a failed package install, up to 500 characters).
        code: The kind of failure (ErrorCode); one of the class's `codes`.
        action: What the caller does next (ErrorAction), from the class.
        hint: What to change, or None. It is also a note of the exception.
        instance_id: The saved instance the call was about, or None.
        context: Structured values for logs.
    """

    action: ClassVar[ErrorAction]
    codes: ClassVar[frozenset[ErrorCode]] = frozenset()

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        context: dict[str, Any] | None,
        hint: str | None,
        instance_id: str | None,
    ) -> None:
        if code not in type(self).codes:  # one action per code, checked at the raise
            raise TypeError(f"{code} is not a {type(self).__name__} code")
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or {}
        self.hint = hint
        self.instance_id = instance_id
        if hint is not None:
            self.add_note(f"Hint: {hint}")

    def __reduce__(self) -> tuple[Any, ...]:
        # Default pickling calls cls(*self.args), which fails on the keyword-only
        # parameters. Rebuild without __init__; the attributes come back from __dict__.
        return (_rebuild_error, (type(self), self.args), self.__dict__)


def _rebuild_error[E: BaseException](cls: type[E], args: tuple[Any, ...]) -> E:
    """Unpickle an error without calling its __init__."""
    return cls.__new__(cls, *args)


class TransientError(SandboxError):
    """The call did nothing you rely on (an exec did not send the code).

    Repeat it unchanged, on this worker or another: it can succeed later. Bound the
    repeats (for example 3, with backoff): the library does not. If the failure closed the session
    (`session.closed`), the work since its last save is lost (all of it, for a
    session never saved): open a new session (new() or reuse()), do that work
    again, then repeat the call.
    """

    action = ErrorAction.RETRY
    codes = frozenset(
        {
            ErrorCode.VM_BOOT_TIMEOUT,
            ErrorCode.QEMU_CRASHED,
            ErrorCode.OVERLAY_FAILED,
            ErrorCode.GVPROXY_FAILED,
            ErrorCode.CAPACITY,
            ErrorCode.GUEST_UNAVAILABLE,
            ErrorCode.GUEST_SILENT_AFTER_RESTORE,
            ErrorCode.LOCAL_DISK,
            ErrorCode.PACKAGE_DOWNLOAD,
            ErrorCode.MIGRATION_FAILED,
            ErrorCode.ASSET_DOWNLOAD,
            ErrorCode.INSTANCE_STORE_UNAVAILABLE,
        }
    )


class OutcomeUnknownError(SandboxError):
    """You do not know the state the call left, or must read it before you do the work again.

    GUEST_OUTCOME_UNKNOWN: the code was sent and then the transport was lost; the session
    is closed. INSTANCE_SAVE_OUTCOME_UNKNOWN: the store gave no clear answer to a save;
    `context["generation"]` and `context["commit_id"]` name the commit it would make: it
    was stored if `describe_instance(id).recent_commits` holds it. INSTANCE_CONFLICT:
    another save was stored first, or the id of a first save is taken; this save was not
    stored. After either save code the session stays open.
    """

    action = ErrorAction.VERIFY
    codes = frozenset(
        {
            ErrorCode.GUEST_OUTCOME_UNKNOWN,
            ErrorCode.INSTANCE_CONFLICT,
            ErrorCode.INSTANCE_SAVE_OUTCOME_UNKNOWN,
        }
    )


class SessionClosedError(SandboxError):
    """The session is closed (close(), idle timeout, or an earlier failure).

    Every later call on it (exec, write_file, read_file, list_files, save)
    raises this error first, whatever its arguments: no argument check runs and
    no local file is opened. One exception: save() on a worker with no instance
    store raises ConfigError (INSTANCE_CONFIG) first. Open a new session with
    new() or reuse().
    """

    action = ErrorAction.RESTART
    codes = frozenset({ErrorCode.SESSION_CLOSED})


class InvalidInputError(SandboxError):
    """The call is wrong: an argument breaks a rule, or the call comes when the
    scheduler cannot take it. The same call fails again. A session stays open."""

    action = ErrorAction.FIX_INPUT
    codes = frozenset(
        {
            ErrorCode.CODE_INVALID,
            ErrorCode.ENV_VAR_INVALID,
            ErrorCode.OUTPUT_LIMIT,
            ErrorCode.PATH_INVALID,
            ErrorCode.GUEST_IO,
            ErrorCode.ARGUMENT_INVALID,
            ErrorCode.PACKAGE_NOT_ALLOWED,
            ErrorCode.PACKAGE_INSTALL_FAILED,
            ErrorCode.INSTANCE_INVALID_ID,
            ErrorCode.SCHEDULER_STATE,
        }
    )


class ConfigError(SandboxError):
    """This worker's setup is wrong: settings, keyring, store, installed binaries,
    images, Python extras, permissions or OS. Change it; the same call fails again."""

    action = ErrorAction.FIX_CONFIG
    codes = frozenset(
        {
            ErrorCode.HOST_DEPENDENCY,
            ErrorCode.EXTRA_MISSING,
            ErrorCode.SETTING_INVALID,
            ErrorCode.ASSET_NOT_FOUND,
            ErrorCode.IMAGE_DAMAGED,
            ErrorCode.PLATFORM_UNSUPPORTED,
            ErrorCode.MIGRATION_UNSUPPORTED,
            ErrorCode.INSTANCE_CONFIG,
            ErrorCode.INSTANCE_KEY_MISSING,
        }
    )


class RouteRule(StrEnum):
    """How a worker satisfies one key of RouteError.required (RouteKey.rule)."""

    EQUAL = "equal"
    """Its value is the same string."""
    AT_LEAST = "at_least"
    """Its value is not below the required one: a number (memory_mb, cpu), or a version
    compared part by part (qemu_version)."""
    CONTAINS = "contains"
    """It holds the required value among its own (an image on its disk)."""


class RouteKey(StrEnum):
    """Keys of RouteError.required and RouteError.found. A release adds keys; it never
    renames or removes one, and a key keeps its rule and its value form."""

    ARCH = "arch"
    """x86_64 or aarch64."""
    ACCELERATOR = "accelerator"
    """kvm, hvf or tcg."""
    CPU_VENDOR = "cpu_vendor"
    """x86 vendor of the guest CPU model; absent on ARM."""
    CPU_MODEL = "cpu_model"
    """The guest -cpu value."""
    QEMU_VERSION = "qemu_version"
    """MAJOR.MINOR.PATCH. QEMU loads a state that the same or an older QEMU saved, with the
    same versioned machine type: a worker whose QEMU is this version or newer."""
    MACHINE_TYPE = "machine_type"
    """A QEMU machine type as QEMU resolves it (virt-11.1, microvm): the worker's QEMU has it."""
    BASE_IMAGE = "base_image"
    """Full SHA-256 of a guest base image the worker can use."""
    PACKAGES_IMAGE = "packages_image"
    """Full SHA-256 of a packages (L2) image the worker can use."""
    MEMORY_MB = "memory_mb"
    """Whole MB of admission budget, the VM overhead included."""
    CPU = "cpu"
    """Cores of admission budget (two decimals), the VM overhead included."""

    @property
    def rule(self) -> RouteRule:
        return _ROUTE_RULES[self]

    def holds(self, have: str | frozenset[str] | None, need: str) -> bool:
        """Whether a worker's value `have` gives `need` by this key's rule: EQUAL, the same
        string; CONTAINS, the value itself or a set that holds it; AT_LEAST, not below
        (qemu_version part by part, the others as numbers), a set never. None: no value, so
        False, but for memory_mb and cpu, where None is no limit (a worker always has a QEMU
        version)."""
        match self.rule:
            case RouteRule.EQUAL:
                return have == need
            case RouteRule.CONTAINS:
                return need in have if isinstance(have, frozenset) else have == need
            case RouteRule.AT_LEAST:
                if have is None:
                    return self is not RouteKey.QEMU_VERSION  # memory_mb, cpu: no limit
                if isinstance(have, frozenset):
                    return False
                if self is RouteKey.QEMU_VERSION:
                    return tuple(map(int, have.split("."))) >= tuple(map(int, need.split(".")))
                return float(have) >= float(need)


_ROUTE_RULES = {
    RouteKey.ARCH: RouteRule.EQUAL,
    RouteKey.ACCELERATOR: RouteRule.EQUAL,
    RouteKey.CPU_VENDOR: RouteRule.EQUAL,
    RouteKey.CPU_MODEL: RouteRule.EQUAL,
    RouteKey.QEMU_VERSION: RouteRule.AT_LEAST,
    RouteKey.MACHINE_TYPE: RouteRule.CONTAINS,
    RouteKey.BASE_IMAGE: RouteRule.CONTAINS,
    RouteKey.PACKAGES_IMAGE: RouteRule.CONTAINS,
    RouteKey.MEMORY_MB: RouteRule.AT_LEAST,
    RouteKey.CPU: RouteRule.AT_LEAST,
}


class RouteError(SandboxError):
    """This worker cannot serve the call; a worker that satisfies `required` can.

    `required` maps RouteKey to the value the call needs: every fact the call checked
    up to this error, not only the facts that differ here. `found` maps this worker's
    values for those keys, and may hold more keys; None: this worker has no value for
    the key (no such image or machine type; no cpu_vendor on ARM; for memory_mb and
    cpu, no limit). Values are plain strings: qemu_version MAJOR.MINOR.PATCH, full
    SHA-256 digests, memory_mb a whole number, cpu with two decimals. A fact that does
    not apply to the saved host (cpu_vendor on ARM) is absent from `required`.

    A worker satisfies `required` when each key holds by its rule (RouteKey.rule):
    EQUAL, its value is the same string; AT_LEAST, its value is not below (qemu_version
    part by part); CONTAINS, it holds the value. A worker that satisfies `required` passes the checks that
    raised this error; a later check of the same call can raise RouteError with more
    keys (a load checks the host, then the budget, then the guest images, and QEMU
    decides at the restore). An empty `required` (data this release cannot read: another
    release wrote it): a worker of another release. Never delete a saved instance for
    this error.
    """

    action = ErrorAction.ROUTE
    codes = frozenset(
        {
            ErrorCode.INSTANCE_HOST_MISMATCH,
            ErrorCode.CAPACITY_NEVER_FITS,
            ErrorCode.MIGRATION_REJECTED,
        }
    )

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        required: Mapping[RouteKey, str],
        found: Mapping[RouteKey, str | None],
        context: dict[str, Any] | None,
        hint: str | None,
        instance_id: str | None,
    ) -> None:
        super().__init__(message, code=code, context=context, hint=hint, instance_id=instance_id)
        self.required: dict[RouteKey, str] = dict(required)
        self.found: dict[RouteKey, str | None] = dict(found)


class InstanceGoneError(SandboxError):
    """The saved instance cannot be used again (deleted, expired, or its stored
    data is damaged). Start a new one."""

    action = ErrorAction.RECREATE
    codes = frozenset(
        {
            ErrorCode.INSTANCE_NOT_FOUND,
            ErrorCode.INSTANCE_EXPIRED,
            ErrorCode.INSTANCE_INTEGRITY,
        }
    )


class InternalError(SandboxError):
    """A bug in exec-sandbox or a broken host: QEMU process cleanup not confirmed,
    a guest that breaks the protocol, a broken invariant. Report it with the log."""

    action = ErrorAction.REPORT
    codes = frozenset(
        {
            ErrorCode.CLEANUP_UNCONFIRMED,
            ErrorCode.GUEST_PROTOCOL,
            ErrorCode.INTERNAL,
        }
    )


# =============================================================================
# Helpers (not exported)
# =============================================================================


def no_instance_store(*, instance_id: str | None) -> ConfigError:
    """The ConfigError of a saved-instance call on a worker without an instance store."""
    return ConfigError(
        "Saved instances need an instance store, and none is configured.",
        code=ErrorCode.INSTANCE_CONFIG,
        context=None,
        hint="Set instance_store and instance_keyring "
        "(or EXEC_SANDBOX_INSTANCE_STORE and EXEC_SANDBOX_INSTANCE_KEYRING); "
        "instance_store=None set in code turns saved instances off, whatever the variables say.",
        instance_id=instance_id,
    )


@contextlib.contextmanager
def named_instance(instance_id: str) -> Generator[None]:
    """Name `instance_id` in each SandboxError raised in the block that names no instance:
    store, VM and guest errors do not know which instance they served."""
    try:
        yield
    except SandboxError as e:
        if e.instance_id is None:
            e.instance_id = instance_id
        raise


def damaged_instance_data(message: str, *, instance_id: str | None) -> InstanceGoneError:
    """The InstanceGoneError of stored data that fails its checks (authentication,
    bounds, size): changed, damaged or truncated."""
    return InstanceGoneError(
        message,
        code=ErrorCode.INSTANCE_INTEGRITY,
        context=None,
        hint="The stored data was changed, damaged or truncated. Start again with a new instance.",
        instance_id=instance_id,
    )


def spawn_error(
    error: OSError, *, binary: str, code: ErrorCode, context: dict[str, Any], hint: str | None
) -> TransientError | ConfigError:
    """The error of a binary that did not start (an OSError of its fork or exec).

    EAGAIN, ENOMEM, EMFILE or ENFILE is a process, memory or file-handle limit that can pass
    (as for a local file: LOCAL_DISK): TransientError `code`.
    Any other errno fails the same way on every retry (ENOENT: missing, EACCES: this user
    cannot run it, ENOEXEC: another CPU architecture): ConfigError HOST_DEPENDENCY, with
    `hint` (how to install the binary).
    """
    context = {**context, "binary": binary, "errno": error.errno}
    message = f"Failed to start {binary}: {error}"
    if error.errno in (errno.EAGAIN, errno.ENOMEM, errno.EMFILE, errno.ENFILE):
        return TransientError(message, code=code, context=context, hint=None, instance_id=None)
    return ConfigError(message, code=ErrorCode.HOST_DEPENDENCY, context=context, hint=hint, instance_id=None)


def host_mismatch(
    *,
    instance_id: str,
    required: Mapping[RouteKey, str],
    found: Mapping[RouteKey, str | None],
    reason: str | None,
) -> RouteError:
    """The RouteError of a worker that cannot run the instance. `required` is the whole
    target, `found` this worker's values for its keys; the message names the keys whose
    value does not hold by its rule (RouteKey.holds), values in JSON form (a cpu_model holds commas). `reason`: why, when the
    values alone do not say it (an image missing from the store too)."""
    differ = [key for key in required if not key.holds(found.get(key), required[key])] or list(required)
    wanted = ", ".join(f"{key}={json.dumps(required[key])}" for key in differ)
    local = ", ".join(f"{key}={json.dumps(found.get(key))}" for key in differ)
    if required:
        message = f"Instance {instance_id} needs {wanted}; this worker has {local}" + (
            f": {reason}." if reason else "."
        )
    else:  # nothing names the target
        message = f"Instance {instance_id}: {reason or 'this worker cannot serve it'}."
    return RouteError(
        message,
        code=ErrorCode.INSTANCE_HOST_MISMATCH,
        required=required,
        found=found,
        context=None if reason is None else {"reason": reason},
        hint=(
            "Reuse it on a worker that satisfies the error's `required` field."
            if required
            else "Reuse it on a worker of another exec-sandbox release."
        )
        + " Keep the instance: its data is intact.",
        instance_id=instance_id,
    )


# =============================================================================
# Internal transport errors (not SandboxError, not exported)
# =============================================================================
# The guest readiness retry and the reconnect probe retry on these alone; the
# callers turn them into a TransientError or an OutcomeUnknownError. As a
# TransientError they would put QEMU_CRASHED into the readiness retry set.


class CommunicationError(Exception):
    """Internal: the transport to the guest or to QEMU failed before the call was sent."""

    def __init__(self, message: str, *, context: dict[str, Any] | None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


class SocketAuthError(CommunicationError):
    """Internal: socket peer authentication failed.

    The Unix socket server does not run as the expected user. Causes: QEMU
    stopped and another process bound the socket path, a race during socket
    creation, or a process that tries to take over the socket.
    """

    def __init__(
        self,
        message: str,
        *,
        expected_uid: int,
        actual_uid: int,
        context: dict[str, Any] | None,
    ) -> None:
        # A new dict: the caller's context stays as it was
        super().__init__(message, context={**(context or {}), "expected_uid": expected_uid, "actual_uid": actual_uid})


class InstanceNotSavedWarning(UserWarning):
    """A session of a saved instance closed cleanly after exec() or write_file()
    calls that no save() followed: those changes are not in the store. Not raised once
    another save of the instance won: no plain save of the session can store them."""

"""Exception hierarchy for exec-sandbox.

All exceptions inherit from SandboxError base class.

Hierarchy:
    SandboxError (base)
    ├── TransientError (retryable marker base)
    │   ├── VmTransientError
    │   │   ├── VmBootTimeoutError     ← guest agent not ready
    │   │   ├── VmOverlayError         ← overlay creation failed
    │   │   ├── VmQemuCrashError       ← QEMU crashed on startup
    │   │   ├── VmGvproxyError         ← gvproxy startup issues
    │   │   └── VmCapacityError        ← pool full (temporary)
    │   ├── PackageInstallTransientError ← transient network during install
    │   ├── BalloonTransientError      ← balloon operations
    │   ├── MigrationTransientError    ← memory snapshot migration
    │   └── CommunicationTransientError ← socket/network transient issues
    ├── PermanentError (non-retryable marker base)
    │   ├── VmPermanentError
    │   │   ├── VmConfigError          ← invalid configuration
    │   │   └── VmDependencyError      ← missing binary/image
    │   └── SessionClosedError         ← session already closed
    ├── InputValidationError (caller-bug marker base)
    │   ├── CodeValidationError       ← empty/null-byte code
    │   └── EnvVarValidationError     ← control chars, size limits
    ├── OutputLimitError              ← stdout/stderr exceeded guest limits
    └── ... (other existing exceptions)

Backward Compatibility:
    VmError = VmPermanentError
    VmTimeoutError = VmBootTimeoutError
    VmBootError = VmTransientError
    QemuImgError = VmOverlayError
    QemuStorageDaemonError = VmOverlayError
    BalloonError = BalloonTransientError
    MigrationError = MigrationTransientError
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuDiagnostics


class SandboxError(Exception):
    """Base exception for all sandbox errors with structured context.

    All custom exceptions in this module inherit from this base class,
    allowing callers to catch any sandbox-related error with a single handler.

    Attributes:
        message: Human-readable error message
        context: Dictionary of structured error context for logging/debugging
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class SandboxDependencyError(SandboxError):
    """Optional dependency missing.

    Raised when an optional dependency is required but not installed.
    For example, aioboto3 is required for S3 snapshot backup.
    """


# =============================================================================
# Transient vs Permanent Error Base Classes
# =============================================================================


class TransientError(SandboxError):
    """Base for transient errors that may succeed on retry.

    Use this as a marker base class to identify errors that are
    potentially recoverable through retry (e.g., resource contention,
    temporary network issues, CPU overload).
    """


class PermanentError(SandboxError):
    """Base for permanent errors that won't succeed on retry.

    Use this as a marker base class to identify errors that are
    not recoverable through retry (e.g., configuration errors,
    missing dependencies, capacity limits).
    """


# =============================================================================
# VM Transient Errors (retryable)
# =============================================================================


class VmTransientError(TransientError):
    """Transient VM errors - may succeed on retry.

    Base class for VM errors that are potentially recoverable,
    such as resource contention, CPU overload, or transient failures.
    """


class VmBootTimeoutError(VmTransientError):
    """VM boot timed out - may succeed under lower load.

    Raised when the guest agent doesn't become ready within the timeout.
    This is often caused by CPU contention and may succeed on retry.
    """


class VmOverlayError(VmTransientError):
    """Overlay creation failed - transient resource issue.

    Raised when qemu-img or qemu-storage-daemon fails to create an overlay.
    Absorbs both QemuImgError and QemuStorageDaemonError for unified handling.

    Attributes:
        stderr: Standard error output from qemu-img (if available)
        error_class: QMP error class from qemu-storage-daemon (if available)
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        stderr: str = "",
        error_class: str | None = None,
    ):
        super().__init__(message, context)
        self.stderr = stderr
        self.error_class = error_class


class VmQemuCrashError(VmTransientError):
    """QEMU crashed during startup - CPU contention.

    Raised when QEMU exits unexpectedly during boot. This is often
    caused by resource pressure and may succeed on retry.

    Attributes:
        diagnostics: Optional QemuDiagnostics with crash context.
            When provided and no explicit context is given,
            context is auto-populated via dataclasses.asdict().
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        *,
        diagnostics: QemuDiagnostics | None = None,
    ):
        if diagnostics is not None and context is None:
            from dataclasses import asdict  # noqa: PLC0415

            context = asdict(diagnostics)
        super().__init__(message, context)
        self.diagnostics = diagnostics


class VmGvproxyError(VmTransientError):
    """gvproxy startup/socket issues.

    Raised when gvproxy fails to start or create its socket.
    May be transient due to resource contention.
    """


# =============================================================================
# VM Permanent Errors (non-retryable)
# =============================================================================


class VmPermanentError(PermanentError):
    """Permanent VM errors - won't succeed on retry.

    Base class for VM errors that are not recoverable through retry,
    such as configuration errors or missing dependencies.
    """


class VmConfigError(VmPermanentError):
    """Invalid VM configuration.

    Raised when VM configuration is invalid (e.g., mutually exclusive
    options, invalid parameters).
    """


class VmCapacityError(VmTransientError):
    """VM pool at capacity.

    Raised when the VM pool is full and cannot accept new VMs.
    This is a transient error - with exponential backoff, capacity
    may become available as other VMs complete and are destroyed.
    """


class VmDependencyError(VmPermanentError):
    """Required dependency missing.

    Raised when a required binary, image, or system user is not available.
    This is a permanent error that requires system configuration changes.
    """


# =============================================================================
# Communication Errors
# =============================================================================


class PackageInstallTransientError(TransientError):
    """Package installation failed due to transient network error — may succeed on retry."""


class BalloonTransientError(TransientError):
    """Balloon operation failed - may succeed on retry.

    Raised when balloon memory control operations fail.
    These are often transient and may succeed on retry.
    """


class MigrationTransientError(TransientError):
    """Migration (memory snapshot) operation failed - may succeed on retry.

    Raised when QEMU migration operations (save/restore memory snapshots) fail.
    These are often transient due to QEMU state or resource contention.
    """


# =============================================================================
# Backward Compatibility Aliases (Public API)
# =============================================================================

# These aliases maintain backward compatibility with the public API.
# Old code using these names will continue to work.
VmError = VmPermanentError
VmTimeoutError = VmBootTimeoutError
VmBootError = VmTransientError

# Internal aliases for import compatibility in other modules.
# These allow existing code to import QemuImgError, etc. from exceptions.
QemuImgError = VmOverlayError
QemuStorageDaemonError = VmOverlayError
BalloonError = BalloonTransientError
MigrationError = MigrationTransientError


class SnapshotError(SandboxError):
    """Snapshot operation failed.

    Raised when creating, loading, or managing VM snapshots encounters an error,
    including filesystem operations or snapshot state corruption.
    """


class CommunicationError(SandboxError):
    """Guest communication failed.

    Raised when communication with the guest VM fails, including
    connection errors, protocol errors, or guest agent unavailability.
    """


class SocketAuthError(CommunicationError):
    """Socket peer authentication failed.

    Raised when Unix socket server is not running as expected user.
    This could indicate:
    - QEMU crashed and another process bound the socket path
    - Race condition during socket creation
    - Malicious process attempting socket hijacking

    Attributes:
        expected_uid: Expected user ID
        actual_uid: Actual user ID from peer credentials
    """

    def __init__(
        self,
        message: str,
        expected_uid: int,
        actual_uid: int,
        context: dict[str, Any] | None = None,
    ):
        ctx = context or {}
        ctx.update({"expected_uid": expected_uid, "actual_uid": actual_uid})
        super().__init__(message, ctx)
        self.expected_uid = expected_uid
        self.actual_uid = actual_uid


class GuestAgentError(SandboxError):
    """Guest agent returned error response.

    Indicates the guest agent processed the request but reported failure.
    The error message contains the guest's stderr/message.
    Used for both package installation and code execution failures.
    """

    def __init__(self, message: str, response: dict[str, Any]):
        super().__init__(message, context={"response": response})
        self.response = response


class PackageNotAllowedError(SandboxError):
    """Package not in allowlist.

    Raised when attempting to install a package that is not present in the
    configured allowlist, preventing potentially unsafe package installations.
    """


class InputValidationError(SandboxError):
    """Base for input validation errors (caller bugs, not VM failures).

    These errors mean the caller passed invalid input. The session/VM is
    unaffected and can be reused — the caller should fix their input and retry.
    """


class CodeValidationError(InputValidationError):
    """Code validation failed.

    Raised when the code string is empty, whitespace-only, or contains
    invalid characters (null bytes).
    """


class EnvVarValidationError(InputValidationError):
    """Environment variable validation failed.

    Raised when environment variable names or values contain invalid
    characters (control characters, null bytes) or exceed size limits.
    """


class OutputLimitError(PermanentError):
    """Output size limit exceeded during code execution.

    Raised when stdout or stderr exceeds the guest-enforced size limit
    (stdout: 1 MB, stderr: 100 KB — see guest-agent/src/constants.rs).
    The REPL session is preserved and can be reused — only the current
    execution's output was too large.

    Inherits PermanentError because the output-size exceedance is not
    retryable by re-sending the same code — the caller must reduce their
    output volume. The VM session itself remains healthy and READY for reuse.

    The partial output (up to the limit) is still delivered via streaming
    callbacks before this error is raised.
    """


class SessionClosedError(PermanentError):
    """Raised when attempting to use a session after it has been closed.

    Sessions are closed explicitly via close(), by idle timeout, or
    when the underlying VM fails. Once closed, all subsequent operations
    (exec, write_file, read_file, list_files) will raise this exception.
    """


class AssetError(SandboxError):
    """Base exception for asset-related errors.

    Raised when downloading, verifying, or processing assets fails.
    """


class AssetDownloadError(AssetError):
    """Asset download failed.

    Raised when downloading an asset from GitHub Releases fails after
    all retry attempts are exhausted.
    """


class AssetChecksumError(AssetError):
    """Asset checksum verification failed.

    Raised when the downloaded asset's SHA256 hash does not match
    the expected hash from the GitHub Release API.
    """


class AssetNotFoundError(AssetError):
    """Asset not found.

    Raised when an asset is not found in the registry or when
    a GitHub Release does not exist for the specified version.
    """

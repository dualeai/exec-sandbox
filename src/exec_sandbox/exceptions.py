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
    │   ├── BalloonTransientError      ← balloon operations
    │   └── CommunicationTransientError ← socket/network transient issues
    ├── PermanentError (non-retryable marker base)
    │   └── VmPermanentError
    │       ├── VmConfigError          ← invalid configuration
    │       └── VmDependencyError      ← missing binary/image
    └── ... (other existing exceptions)

Backward Compatibility:
    VmError = VmPermanentError
    VmTimeoutError = VmBootTimeoutError
    VmBootError = VmTransientError
    QemuImgError = VmOverlayError
    QemuStorageDaemonError = VmOverlayError
    BalloonError = BalloonTransientError
"""

from typing import Any


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
    """


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


class BalloonTransientError(TransientError):
    """Balloon operation failed - may succeed on retry.

    Raised when balloon memory control operations fail.
    These are often transient and may succeed on retry.
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


class SnapshotError(SandboxError):
    """Snapshot operation failed.

    Raised when creating, loading, or managing VM snapshots encounters an error,
    including filesystem operations or snapshot state corruption.
    """


class CommunicationError(SandboxError):
    """Guest communication failed.

    Raised when communication with the guest VM fails, including TCP
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


class EnvVarValidationError(SandboxError):
    """Environment variable validation failed.

    Raised when environment variable names or values contain invalid
    characters (control characters, null bytes) or exceed size limits.
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

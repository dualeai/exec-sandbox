"""Exception hierarchy for exec-sandbox.

All exceptions inherit from SandboxError base class.
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


class VmError(SandboxError):
    """VM operation failed.

    Raised when a virtual machine operation encounters an error during
    execution, including failures in start, stop, or runtime operations.
    """


class VmTimeoutError(VmError):
    """VM operation timed out.

    Raised when a virtual machine operation exceeds its allocated time limit,
    such as boot timeout or execution timeout.
    """


class VmBootError(VmError):
    """VM failed to boot.

    Raised when a virtual machine fails to complete its boot sequence,
    including kernel initialization or guest agent startup failures.
    """


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

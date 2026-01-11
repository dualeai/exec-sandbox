"""Lazy imports with clear error messages for optional dependencies."""

from exec_sandbox.exceptions import SandboxDependencyError


def require_aioboto3():
    """Import aioboto3, raise clear error if not installed.

    Returns:
        The aioboto3 module.

    Raises:
        SandboxDependencyError: If aioboto3 is not installed.
    """
    try:
        import aioboto3  # type: ignore[import-untyped]  # noqa: PLC0415

        return aioboto3
    except ImportError as e:
        raise SandboxDependencyError(
            "S3 snapshot backup requires aioboto3. Install with: pip install exec-sandbox[s3]"
        ) from e

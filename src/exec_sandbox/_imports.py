"""Lazy imports with clear error messages for optional dependencies."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

from exec_sandbox.exceptions import ConfigError, ErrorCode

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Callable

    from exec_sandbox.instance.manager import InstanceAdmin, InstanceManager
    from exec_sandbox.instance.s3_store import S3InstanceStore
    from exec_sandbox.settings import InstanceStoreKind, Settings


def require_aioboto3():
    """Import aioboto3, raise clear error if not installed.

    Returns:
        The aioboto3 module.

    Raises:
        ConfigError: EXTRA_MISSING if aioboto3 is not installed.
    """
    try:
        import aioboto3  # type: ignore[import-untyped]  # noqa: PLC0415

        return aioboto3
    except ImportError as e:
        raise _extra_missing("S3 snapshot backup requires aioboto3. Install with: pip install exec-sandbox[s3]") from e


def require_instance_manager(*, store: InstanceStoreKind) -> type[InstanceManager]:
    """Import the saved-instance manager (cryptography), raise clear error if not installed."""
    try:
        from exec_sandbox.instance.manager import InstanceManager  # noqa: PLC0415

        return InstanceManager
    except ImportError as e:
        raise _instances_need_cryptography(store) from e


def require_instance_admin(
    *, store: InstanceStoreKind
) -> Callable[[Settings], contextlib.AbstractAsyncContextManager[InstanceAdmin]]:
    """Import `open_instance_admin` (describe, list, delete without a Scheduler), raise clear error if not installed."""
    try:
        from exec_sandbox.instance.manager import open_instance_admin  # noqa: PLC0415

        return open_instance_admin
    except ImportError as e:
        raise _instances_need_cryptography(store) from e


def _instances_need_cryptography(store: InstanceStoreKind) -> ConfigError:
    install = "'exec-sandbox[s3]'" if store == "s3" else "'exec-sandbox[cli]' (or 'exec-sandbox[s3]')"
    return _extra_missing(
        f"Saved instances in the {store} store require cryptography. Install with: pip install {install}"
    )


def _extra_missing(message: str) -> ConfigError:
    """The ConfigError of a Python extra that is not installed (the message names the pip command)."""
    return ConfigError(message, code=ErrorCode.EXTRA_MISSING, context=None, hint=None, instance_id=None)


def require_s3_instance_store() -> type[S3InstanceStore]:
    """Import the S3 instance store (aiobotocore), raise clear error if not installed."""
    try:
        from exec_sandbox.instance.s3_store import S3InstanceStore  # noqa: PLC0415

        return S3InstanceStore
    except ImportError as e:
        raise _extra_missing(
            "The S3 instance store requires the s3 extra. Install with: pip install 'exec-sandbox[s3]'"
        ) from e


_CLI_DEPENDENCIES = frozenset({"click", "cryptography", "keyring"})


def require_cli_keyring() -> None:
    """Raise a clear error if cli_keyring cannot be imported (keyring and cryptography, from the cli
    extra); the CLI imports it after this check, only to open the OS keyring: every other start skips it."""
    try:
        importlib.import_module("exec_sandbox.cli_keyring")
    except ImportError as e:
        if e.name not in _CLI_DEPENDENCIES:
            raise
        raise _extra_missing(_cli_needs(e.name)) from e


def cli_main() -> None:
    """Console script entry (sbx, exec-sandbox): the CLI needs the cli extra."""
    try:
        from exec_sandbox.cli import main  # noqa: PLC0415 - optional dependencies
    except ImportError as e:
        if e.name not in _CLI_DEPENDENCIES:
            raise
        sys.exit(_cli_needs(e.name))
    main()


def _cli_needs(module: str) -> str:
    return f"The exec-sandbox CLI needs {module}: pip install 'exec-sandbox[cli]'"

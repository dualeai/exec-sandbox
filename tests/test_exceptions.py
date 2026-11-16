"""Tests for the error classes: one class per action, one class per code."""

import errno
import os
import pickle

import pytest

import exec_sandbox
from exec_sandbox import exceptions
from exec_sandbox.exceptions import (
    CommunicationError,
    ConfigError,
    ErrorAction,
    ErrorCode,
    InstanceGoneError,
    InternalError,
    InvalidInputError,
    OutcomeUnknownError,
    RouteError,
    RouteKey,
    RouteRule,
    SandboxError,
    SessionClosedError,
    SocketAuthError,
    TransientError,
)

ACTION_CLASSES: list[type[SandboxError]] = [
    TransientError,
    OutcomeUnknownError,
    SessionClosedError,
    InvalidInputError,
    ConfigError,
    RouteError,
    InstanceGoneError,
    InternalError,
]

PUBLIC_ERROR_NAMES = {
    "SandboxError",
    "TransientError",
    "OutcomeUnknownError",
    "SessionClosedError",
    "InvalidInputError",
    "ConfigError",
    "RouteError",
    "RouteKey",
    "RouteRule",
    "ROUTE_ANY",
    "InstanceGoneError",
    "InternalError",
    "ErrorAction",
    "ErrorCode",
    "InstanceNotSavedWarning",
}


def _make(cls: type[SandboxError], code: ErrorCode) -> SandboxError:
    if cls is RouteError:
        return RouteError(
            "needs another worker",
            code=code,
            required={"arch": "aarch64"},
            found={"arch": "x86_64"},
            context={"vm_id": "vm-1"},
            hint="do this",
            instance_id="0" * 32,
        )
    return cls("failed", code=code, context={"vm_id": "vm-1"}, hint="do this", instance_id="0" * 32)


def _first_code(cls: type[SandboxError]) -> ErrorCode:
    return min(cls.codes)


class TestCodes:
    """Each code belongs to exactly one class, and each class has one action."""

    def test_codes_partition_error_code(self) -> None:
        seen: set[ErrorCode] = set()
        for cls in ACTION_CLASSES:
            assert cls.codes, cls.__name__
            assert not seen & cls.codes, f"{cls.__name__} shares codes"
            seen |= cls.codes
        assert seen == set(ErrorCode)

    def test_each_action_belongs_to_one_class(self) -> None:
        actions = [cls.action for cls in ACTION_CLASSES]
        assert sorted(actions) == sorted(ErrorAction)

    def test_values_equal_names(self) -> None:
        assert all(member.value == member.name for member in ErrorCode)
        assert all(member.value == member.name for member in ErrorAction)

    def test_every_class_is_a_direct_sandbox_error(self) -> None:
        for cls in ACTION_CLASSES:
            assert cls.__bases__ == (SandboxError,), cls.__name__

    @pytest.mark.parametrize("cls", ACTION_CLASSES, ids=lambda c: c.__name__)
    def test_code_of_another_class_raises_type_error(self, cls: type[SandboxError]) -> None:
        other = next(code for code in ErrorCode if code not in cls.codes)
        with pytest.raises(TypeError, match=f"{other} is not a {cls.__name__} code"):
            _make(cls, other)

    def test_base_class_is_never_raised_itself(self) -> None:
        with pytest.raises(TypeError, match="is not a SandboxError code"):
            SandboxError("failed", code=ErrorCode.INTERNAL, context=None, hint=None, instance_id=None)

    @pytest.mark.parametrize("cls", ACTION_CLASSES, ids=lambda c: c.__name__)
    def test_fields(self, cls: type[SandboxError]) -> None:
        code = _first_code(cls)
        error = _make(cls, code)
        assert error.code is code
        assert error.action is cls.action
        assert error.instance_id == "0" * 32
        assert error.context == {"vm_id": "vm-1"}
        assert isinstance(error, SandboxError)

    def test_context_none_gives_empty_dict(self) -> None:
        error = TransientError("failed", code=ErrorCode.CAPACITY, context=None, hint=None, instance_id=None)
        assert error.context == {}


class TestMessageAndHint:
    """`str(error)` is one line; the hint is a PEP 678 note."""

    def test_str_is_the_message(self) -> None:
        error = ConfigError(
            "bad setting", code=ErrorCode.SETTING_INVALID, context=None, hint="fix it", instance_id=None
        )
        assert str(error) == "bad setting"
        assert error.message == "bad setting"

    def test_hint_is_a_note(self) -> None:
        error = ConfigError(
            "bad setting", code=ErrorCode.SETTING_INVALID, context=None, hint="fix it", instance_id=None
        )
        assert error.hint == "fix it"
        assert error.__notes__ == ["Hint: fix it"]

    def test_no_hint_no_note(self) -> None:
        error = InternalError("bug", code=ErrorCode.INTERNAL, context=None, hint=None, instance_id=None)
        assert error.hint is None
        assert not hasattr(error, "__notes__")


class TestPickle:
    """Errors cross process boundaries (multiprocessing, executors) intact."""

    @pytest.mark.parametrize("cls", ACTION_CLASSES, ids=lambda c: c.__name__)
    def test_round_trip(self, cls: type[SandboxError]) -> None:
        error = _make(cls, _first_code(cls))
        restored = pickle.loads(pickle.dumps(error))  # noqa: S301 - bytes produced by this test
        assert type(restored) is type(error)
        assert restored.args == error.args
        assert str(restored) == str(error)
        assert vars(restored) == vars(error)
        assert restored.__notes__ == error.__notes__

    def test_route_error_keeps_required_and_found(self) -> None:
        error = _make(RouteError, ErrorCode.INSTANCE_HOST_MISMATCH)
        assert isinstance(error, RouteError)
        restored = pickle.loads(pickle.dumps(error))  # noqa: S301 - bytes produced by this test
        assert restored.required == {"arch": "aarch64"}
        assert restored.found == {"arch": "x86_64"}


class TestHelpers:
    """The helpers build the right class and code."""

    def test_host_mismatch_names_required_values(self) -> None:
        """`required` keeps the whole target; the message names only the values that differ,
        in JSON form (a cpu_model holds commas; None is null)."""
        required = {RouteKey.ARCH: "x86_64", RouteKey.CPU_MODEL: "m,+tsc-deadline", RouteKey.COUNTER_HZ: "2700000000"}
        found: dict[RouteKey, str | None] = {
            RouteKey.ARCH: "aarch64",
            RouteKey.CPU_MODEL: "host",
            RouteKey.COUNTER_HZ: None,
        }
        error = exceptions.host_mismatch(instance_id="0" * 32, required=required, found=found, reason=None)
        assert str(error) == (
            f'Instance {"0" * 32} needs arch="x86_64", cpu_model="m,+tsc-deadline", counter_hz="2700000000"; '
            'this worker has arch="aarch64", cpu_model="host", counter_hz=null.'
        )
        assert type(error) is RouteError
        assert error.code is ErrorCode.INSTANCE_HOST_MISMATCH
        assert error.action == "ROUTE"
        assert (error.required, error.found, error.context) == (required, found, {})
        assert error.hint == (
            "Reuse it on a worker that satisfies the error's `required` field. Keep the instance: its data is intact."
        )

    def test_host_mismatch_names_only_what_differs_and_the_reason(self) -> None:
        error = exceptions.host_mismatch(
            instance_id="vm",
            required={RouteKey.ARCH: "aarch64", RouteKey.BASE_IMAGE: "a" * 64},
            found={RouteKey.ARCH: "aarch64", RouteKey.BASE_IMAGE: None},
            reason="the image is not on this worker and not in the store",
        )
        assert str(error) == (
            f'Instance vm needs base_image="{"a" * 64}"; this worker has base_image=null: '
            "the image is not on this worker and not in the store."
        )
        assert error.context == {"reason": "the image is not on this worker and not in the store"}
        nothing = exceptions.host_mismatch(
            instance_id="vm",
            required={},
            found={},
            reason="this release cannot read its block",
        )
        assert str(nothing) == "Instance vm: this release cannot read its block."
        assert nothing.hint == (
            "Reuse it on a worker of another exec-sandbox release. Keep the instance: its data is intact."
        )

    def test_host_mismatch_names_a_key_by_its_rule(self) -> None:
        """A budget at or above the need, or with no limit (None), and a machine type this
        worker has pass: the message names only the image it lacks."""
        error = exceptions.host_mismatch(
            instance_id="vm",
            required={
                RouteKey.MEMORY_MB: "272",
                RouteKey.CPU: "1.25",
                RouteKey.MACHINE_TYPE: "virt-11.1",
                RouteKey.BASE_IMAGE: "a" * 64,
            },
            found={
                RouteKey.MEMORY_MB: "8000",
                RouteKey.CPU: None,
                RouteKey.MACHINE_TYPE: "virt-11.1",
                RouteKey.BASE_IMAGE: None,
            },
            reason=None,
        )
        assert str(error) == f'Instance vm needs base_image="{"a" * 64}"; this worker has base_image=null.'

    @pytest.mark.parametrize(
        ("key", "have", "need", "holds"),
        [
            pytest.param(RouteKey.MEMORY_MB, None, "272", True, id="no-memory-limit"),
            pytest.param(RouteKey.QEMU_VERSION, None, "11.1.3", False, id="no-qemu-version"),
            pytest.param(RouteKey.CPU, frozenset({"4.00"}), "1.25", False, id="at-least-with-a-set"),
            pytest.param(RouteKey.BASE_IMAGE, None, "a" * 64, False, id="no-image"),
            pytest.param(RouteKey.BASE_IMAGE, "a" * 64, "a" * 64, True, id="contains-the-value"),
            pytest.param(RouteKey.ARCH, None, "aarch64", False, id="equal-no-value"),
            pytest.param(RouteKey.COUNTER_HZ, "24000000", "24000000", True, id="same-rate"),
            pytest.param(RouteKey.COUNTER_HZ, "1000000000", "24000000", False, id="other-rate"),
            pytest.param(RouteKey.COUNTER_HZ, "any", "2700000000", True, id="any-rate"),
            pytest.param(RouteKey.COUNTER_HZ, None, "2700000000", False, id="no-exact-rate"),
        ],
    )
    def test_holds_by_rule(self, key: RouteKey, have: str | frozenset[str] | None, need: str, holds: bool) -> None:
        assert key.holds(have, need) is holds

    def test_every_route_key_has_its_rule(self) -> None:
        """The rule of each key is part of the contract: a broker compares by it."""
        assert {key: key.rule for key in RouteKey} == {
            RouteKey.ARCH: RouteRule.EQUAL,
            RouteKey.ACCELERATOR: RouteRule.EQUAL,
            RouteKey.CPU_MODEL: RouteRule.EQUAL,
            RouteKey.CPU_ID: RouteRule.EQUAL,
            RouteKey.COUNTER_HZ: RouteRule.EQUAL_OR_ANY,
            RouteKey.QEMU_VERSION: RouteRule.AT_LEAST,
            RouteKey.MACHINE_TYPE: RouteRule.CONTAINS,
            RouteKey.BASE_IMAGE: RouteRule.CONTAINS,
            RouteKey.PACKAGES_IMAGE: RouteRule.CONTAINS,
            RouteKey.MEMORY_MB: RouteRule.AT_LEAST,
            RouteKey.CPU: RouteRule.AT_LEAST,
        }

    def test_no_instance_store(self) -> None:
        """The hint names both ways to set a store, the settings and their variables, and that
        instance_store=None in code wins over a variable that is set."""
        error = exceptions.no_instance_store(instance_id="abc")
        assert type(error) is ConfigError
        assert error.code is ErrorCode.INSTANCE_CONFIG
        assert error.instance_id == "abc"
        assert str(error) == "Saved instances need an instance store, and none is configured."
        assert error.hint == (
            "Set instance_store and instance_keyring "
            "(or EXEC_SANDBOX_INSTANCE_STORE and EXEC_SANDBOX_INSTANCE_KEYRING); "
            "instance_store=None set in code turns saved instances off, whatever the variables say."
        )

    def test_damaged_instance_data(self) -> None:
        """The caller's message, and a hint that tells to start again with a new instance."""
        error = exceptions.damaged_instance_data("broken", instance_id="abc")
        assert type(error) is InstanceGoneError
        assert error.code is ErrorCode.INSTANCE_INTEGRITY
        assert (str(error), error.instance_id) == ("broken", "abc")
        assert error.hint == "The stored data was changed, damaged or truncated. Start again with a new instance."

    @pytest.mark.parametrize(
        ("err", "cls", "code", "hint"),
        [
            (errno.EAGAIN, TransientError, ErrorCode.GVPROXY_FAILED, None),
            (errno.ENOMEM, TransientError, ErrorCode.GVPROXY_FAILED, None),
            (errno.EMFILE, TransientError, ErrorCode.GVPROXY_FAILED, None),
            (errno.ENFILE, TransientError, ErrorCode.GVPROXY_FAILED, None),
            (errno.ENOENT, ConfigError, ErrorCode.HOST_DEPENDENCY, "Install it."),
            (errno.EACCES, ConfigError, ErrorCode.HOST_DEPENDENCY, "Install it."),
            (errno.ENOEXEC, ConfigError, ErrorCode.HOST_DEPENDENCY, "Install it."),
            (errno.EPERM, ConfigError, ErrorCode.HOST_DEPENDENCY, "Install it."),
        ],
        ids=[
            "process-limit",
            "no-memory",
            "no-file-handles",
            "no-system-file-handles",
            "missing",
            "no-exec-bit",
            "other-arch",
            "other",
        ],
    )
    def test_spawn_error_by_errno(self, err: int, cls: type[SandboxError], code: ErrorCode, hint: str | None) -> None:
        """A process or memory limit can pass (the site's TransientError, no install hint);
        any other spawn errno fails on every retry: ConfigError HOST_DEPENDENCY with the hint."""
        error = exceptions.spawn_error(
            OSError(err, os.strerror(err)),
            binary="tool",
            code=ErrorCode.GVPROXY_FAILED,
            context={"vm_id": "vm-1"},
            hint="Install it.",
        )
        assert (type(error), error.code, error.hint) == (cls, code, hint)
        assert str(error) == f"Failed to start tool: [Errno {err}] {os.strerror(err)}"
        assert error.context == {"vm_id": "vm-1", "binary": "tool", "errno": err}


class TestInternalClasses:
    """The transport errors stay internal: not SandboxError, not exported."""

    def test_not_sandbox_errors(self) -> None:
        assert not issubclass(CommunicationError, SandboxError)
        assert issubclass(SocketAuthError, CommunicationError)

    def test_socket_auth_error_leaves_the_callers_context_as_it_was(self) -> None:
        """The error's context adds both uids; the dict the caller gave is not changed."""
        given = {"socket": "/run/qmp.sock"}

        error = SocketAuthError("peer uid 0", expected_uid=501, actual_uid=0, context=given)

        assert error.context == {"socket": "/run/qmp.sock", "expected_uid": 501, "actual_uid": 0}
        assert given == {"socket": "/run/qmp.sock"}

    def test_not_exported(self) -> None:
        assert "CommunicationError" not in exec_sandbox.__all__
        assert "SocketAuthError" not in exec_sandbox.__all__


class TestExports:
    """The package exports exactly the 15 error names (RouteKey, RouteRule and ROUTE_ANY included)."""

    def test_all_holds_exactly_the_error_names(self) -> None:
        from_exceptions = {
            name for name in exec_sandbox.__all__ if getattr(exceptions, name, None) is getattr(exec_sandbox, name)
        }
        assert from_exceptions == PUBLIC_ERROR_NAMES

    def test_names_are_the_module_objects(self) -> None:
        for name in PUBLIC_ERROR_NAMES:
            assert getattr(exec_sandbox, name) is getattr(exceptions, name)

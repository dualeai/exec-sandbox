"""Integration tests for environment variable security validation.

Tests the full Python->Rust guest agent flow for env var validation.
These tests require a running VM and verify defense-in-depth.

References:
- OWASP Top 10:2025 A05 Injection
- Terminal escape injection attacks
"""

from typing import TYPE_CHECKING

import pytest

from exec_sandbox import constants
from exec_sandbox.exceptions import ErrorCode, InvalidInputError
from exec_sandbox.guest_agent_protocol import ExecuteCodeRequest, StreamingErrorMessage
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM


class TestEnvVarSecurityIntegration:
    """Integration tests for env var validation across Python->Rust boundary."""

    # =========================================================================
    # Allowed characters (tab, UTF-8); the refused ones: TestGuestAgentControlCharValidation
    # =========================================================================

    async def test_tab_allowed_by_guest(self, dual_scheduler: Scheduler) -> None:
        """Tab character in env var value is allowed."""
        result = await run_once(
            dual_scheduler,
            VmSpec(language=Language.PYTHON),
            "import os; print(repr(os.environ.get('FOO')))",
            env_vars={"FOO": "col1\tcol2"},
        )
        assert result.exit_code == 0
        assert result.stdout == "'col1\\tcol2'\n"

    async def test_utf8_allowed_by_guest(self, dual_scheduler: Scheduler) -> None:
        """UTF-8 characters in env var value are allowed."""
        result = await run_once(
            dual_scheduler,
            VmSpec(language=Language.PYTHON),
            "import os; print(os.environ.get('GREETING'))",
            env_vars={"GREETING": "Hello 世界"},
        )
        assert result.exit_code == 0
        assert result.stdout == "Hello 世界\n"

    # =========================================================================
    # Blocked Environment Variables (Security Blocklist)
    # =========================================================================

    async def test_blocked_env_var_refused_by_guest(self, dual_scheduler: Scheduler) -> None:
        """A blocked name (LD_PRELOAD) reaches the caller as ENV_VAR_INVALID and runs nothing.
        The blocked names, in any case: guest-agent/src/validation.rs tests."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=r"(?i)blocked"
        ):
            await run_once(
                dual_scheduler,
                VmSpec(language=Language.PYTHON),
                "print('should not run')",
                env_vars={"LD_PRELOAD": "/tmp/malicious.so"},
            )

    # =========================================================================
    # Valid Environment Variables (Positive Tests)
    # =========================================================================

    async def test_multiple_env_vars_work(self, dual_scheduler: Scheduler) -> None:
        """Each of three custom env vars reaches the code with its own value (an unset var prints None)."""
        result = await run_once(
            dual_scheduler,
            VmSpec(language=Language.PYTHON),
            """
import os
print(os.environ.get('VAR1'))
print(os.environ.get('VAR2'))
print(os.environ.get('VAR3'))
""",
            env_vars={"VAR1": "one", "VAR2": "two", "VAR3": "three"},
        )
        assert result.exit_code == 0
        assert result.stdout == "one\ntwo\nthree\n"


class TestGuestAgentControlCharValidation:
    """Tests that Rust guest-agent rejects control characters (bypassing Pydantic).

    The host refuses these values first (Pydantic), so the guest check can only be reached on
    the wire: model_construct() builds a request with no validation, and the test reads the
    messages the guest sends back. Each refused character: guest-agent/src/validation.rs
    tests. Tab is allowed on both sides: test_tab_allowed_by_guest reaches the guest through
    the public API.
    """

    async def test_control_character_rejected_by_rust(self, running_vm: "QemuVM") -> None:
        """An escape sequence in a value: the guest replies with one env_var_error and runs
        nothing."""
        await running_vm.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
        request = ExecuteCodeRequest.model_construct(
            action="exec",
            language=Language.PYTHON,
            code="print('should not run')",
            reseed=False,
            timeout=30,
            env_vars={"FOO": "\x1b[31mred\x1b[0m"},
        )
        messages = [msg async for msg in running_vm.channel.stream_messages(request, timeout=30)]

        assert len(messages) == 1, messages
        (reply,) = messages
        assert isinstance(reply, StreamingErrorMessage), reply
        assert (reply.error_type, reply.message) == (
            "env_var_error",
            "Environment variable 'FOO' value contains forbidden control character",
        )

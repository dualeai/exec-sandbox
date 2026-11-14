"""Integration tests for environment variable security validation.

Tests the full Python->Rust guest agent flow for env var validation.
These tests require a running VM and verify defense-in-depth.

References:
- OWASP Top 10:2025 A05 Injection
- Terminal escape injection attacks
"""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest

from exec_sandbox.exceptions import ErrorCode, InvalidInputError
from exec_sandbox.guest_agent_protocol import ExecuteCodeRequest
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_manager import VmManager
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM


class TestEnvVarSecurityIntegration:
    """Integration tests for env var validation across Python->Rust boundary."""

    # =========================================================================
    # Control Character Rejection (Guest Agent Validation)
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
        assert "col1\\tcol2" in result.stdout

    async def test_utf8_allowed_by_guest(self, dual_scheduler: Scheduler) -> None:
        """UTF-8 characters in env var value are allowed."""
        result = await run_once(
            dual_scheduler,
            VmSpec(language=Language.PYTHON),
            "import os; print(os.environ.get('GREETING'))",
            env_vars={"GREETING": "Hello 世界"},
        )
        assert result.exit_code == 0
        assert "世界" in result.stdout

    # =========================================================================
    # Blocked Environment Variables (Security Blocklist)
    # =========================================================================

    async def test_ld_preload_blocked(self, dual_scheduler: Scheduler) -> None:
        """LD_PRELOAD is blocked (arbitrary code execution via library injection)."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=r"(?i)blocked"
        ):
            await run_once(
                dual_scheduler,
                VmSpec(language=Language.PYTHON),
                "print('should not run')",
                env_vars={"LD_PRELOAD": "/tmp/malicious.so"},
            )

    async def test_ld_library_path_blocked(self, dual_scheduler: Scheduler) -> None:
        """LD_LIBRARY_PATH is blocked (library search path manipulation)."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=r"(?i)blocked"
        ):
            await run_once(
                dual_scheduler,
                VmSpec(language=Language.PYTHON),
                "print('should not run')",
                env_vars={"LD_LIBRARY_PATH": "/tmp/malicious"},
            )

    async def test_node_options_blocked(self, dual_scheduler: Scheduler) -> None:
        """NODE_OPTIONS is blocked (Node.js runtime manipulation)."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=r"(?i)blocked"
        ):
            await run_once(
                dual_scheduler,
                VmSpec(language=Language.JAVASCRIPT),
                "console.log('should not run')",
                env_vars={"NODE_OPTIONS": "--expose-gc --max-old-space-size=8192"},
            )

    async def test_pythonstartup_blocked(self, dual_scheduler: Scheduler) -> None:
        """PYTHONSTARTUP is blocked (arbitrary code execution on Python start)."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=r"(?i)blocked"
        ):
            await run_once(
                dual_scheduler,
                VmSpec(language=Language.PYTHON),
                "print('should not run')",
                env_vars={"PYTHONSTARTUP": "/tmp/malicious.py"},
            )

    async def test_bash_env_blocked(self, dual_scheduler: Scheduler) -> None:
        """BASH_ENV is blocked (arbitrary code execution on bash start)."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=r"(?i)blocked"
        ):
            await run_once(
                dual_scheduler,
                VmSpec(language=Language.RAW),
                "echo 'should not run'",
                env_vars={"BASH_ENV": "/tmp/malicious.sh"},
            )

    async def test_path_blocked(self, dual_scheduler: Scheduler) -> None:
        """PATH is blocked (executable search path manipulation)."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=r"(?i)blocked"
        ):
            await run_once(
                dual_scheduler,
                VmSpec(language=Language.PYTHON),
                "print('should not run')",
                env_vars={"PATH": "/tmp/malicious:/usr/bin"},
            )

    async def test_glibc_tunables_blocked(self, dual_scheduler: Scheduler) -> None:
        """GLIBC_TUNABLES is blocked (CVE-2023-4911 mitigation)."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=r"(?i)blocked"
        ):
            await run_once(
                dual_scheduler,
                VmSpec(language=Language.PYTHON),
                "print('should not run')",
                env_vars={"GLIBC_TUNABLES": "glibc.tune.hwcaps=-AVX2"},
            )

    async def test_blocked_env_var_case_insensitive(self, dual_scheduler: Scheduler) -> None:
        """Blocked env var check is case-insensitive."""
        with pytest.raises(
            InvalidInputError, check=lambda e: e.code is ErrorCode.ENV_VAR_INVALID, match=r"(?i)blocked"
        ):
            await run_once(
                dual_scheduler,
                VmSpec(
                    language=Language.PYTHON,
                ),
                "print('should not run')",
                env_vars={"ld_preload": "/tmp/malicious.so"},  # lowercase
            )

    # =========================================================================
    # Valid Environment Variables (Positive Tests)
    # =========================================================================

    async def test_custom_env_var_works(self, dual_scheduler: Scheduler) -> None:
        """Custom env vars are passed to executed code."""
        result = await run_once(
            dual_scheduler,
            VmSpec(language=Language.PYTHON),
            "import os; print(os.environ.get('MY_VAR', 'not set'))",
            env_vars={"MY_VAR": "hello_world"},
        )
        assert result.exit_code == 0
        assert "hello_world" in result.stdout

    async def test_multiple_env_vars_work(self, dual_scheduler: Scheduler) -> None:
        """Multiple custom env vars work correctly."""
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
        assert "one" in result.stdout
        assert "two" in result.stdout
        assert "three" in result.stdout


class TestGuestAgentControlCharValidation:
    """Tests that Rust guest-agent rejects control characters (bypassing Pydantic).

    The host refuses these values first (Pydantic), so the guest check can only be reached on
    the wire: model_construct() builds a request with no validation, and the test reads the
    messages the guest sends back. Tab is allowed on both sides: test_tab_allowed_by_guest
    reaches the guest through the public API.
    """

    @pytest.fixture
    async def running_vm(self, vm_manager: VmManager) -> AsyncGenerator["QemuVM", None]:
        """Create a VM and yield it for testing."""
        from exec_sandbox.qemu_vm import QemuVM  # noqa: TC001

        vm: QemuVM = await vm_manager.create_vm(
            VmSpec(language=Language.PYTHON, memory_mb=256, allow_network=False, allowed_domains=None),
            tenant_id="test",
            task_id="test-control-char",
            cache_leases=(),
        )
        try:
            yield vm
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_control_characters_rejected_by_rust(self, running_vm: "QemuVM") -> None:
        """NUL, ESC, LF, CR, BEL and DEL in a value, and a control character in a name: the guest
        replies with one env_var_error and runs nothing. One VM serves each request in turn
        (an error reply leaves the connection open)."""
        from exec_sandbox import constants
        from exec_sandbox.guest_agent_protocol import StreamingErrorMessage

        cases = {
            "null byte": {"FOO": "val\x00ue"},
            "escape sequence": {"FOO": "\x1b[31mred\x1b[0m"},
            "newline": {"FOO": "line1\nline2"},
            "carriage return": {"FOO": "start\roverwrite"},
            "bell (0x07)": {"FOO": "ding\x07"},
            "DEL (0x7F)": {"FOO": "delete\x7f"},
            "control character in the name": {"F\x01OO": "value"},
        }
        await running_vm.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)
        for case, env_vars in cases.items():
            request = ExecuteCodeRequest.model_construct(
                action="exec",
                language=Language.PYTHON,
                code="print('should not run')",
                timeout=30,
                env_vars=env_vars,
            )
            messages = [msg async for msg in running_vm.channel.stream_messages(request, timeout=30)]
            assert len(messages) == 1, (case, messages)
            (reply,) = messages
            assert isinstance(reply, StreamingErrorMessage), (case, reply)
            assert reply.error_type == "env_var_error", (case, reply)
            assert "contains forbidden control character" in reply.message, (case, reply)

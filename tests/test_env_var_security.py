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

from exec_sandbox.guest_agent_protocol import ExecuteCodeRequest
from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_manager import VmManager

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM


class TestEnvVarSecurityIntegration:
    """Integration tests for env var validation across Python->Rust boundary."""

    # =========================================================================
    # Control Character Rejection (Guest Agent Validation)
    # =========================================================================

    async def test_tab_allowed_by_guest(self, dual_scheduler: Scheduler) -> None:
        """Tab character in env var value is allowed."""
        result = await dual_scheduler.run(
            code="import os; print(repr(os.environ.get('FOO')))",
            language=Language.PYTHON,
            env_vars={"FOO": "col1\tcol2"},
        )
        assert result.exit_code == 0
        assert "col1\\tcol2" in result.stdout

    async def test_utf8_allowed_by_guest(self, dual_scheduler: Scheduler) -> None:
        """UTF-8 characters in env var value are allowed."""
        result = await dual_scheduler.run(
            code="import os; print(os.environ.get('GREETING'))",
            language=Language.PYTHON,
            env_vars={"GREETING": "Hello 世界"},
        )
        assert result.exit_code == 0
        assert "世界" in result.stdout

    # =========================================================================
    # Blocked Environment Variables (Security Blocklist)
    # =========================================================================

    async def test_ld_preload_blocked(self, dual_scheduler: Scheduler) -> None:
        """LD_PRELOAD is blocked (arbitrary code execution via library injection)."""
        result = await dual_scheduler.run(
            code="print('should not run')",
            language=Language.PYTHON,
            env_vars={"LD_PRELOAD": "/tmp/malicious.so"},
        )
        assert result.exit_code != 0
        assert "blocked" in result.stderr.lower() or "LD_PRELOAD" in result.stderr

    async def test_ld_library_path_blocked(self, dual_scheduler: Scheduler) -> None:
        """LD_LIBRARY_PATH is blocked (library search path manipulation)."""
        result = await dual_scheduler.run(
            code="print('should not run')",
            language=Language.PYTHON,
            env_vars={"LD_LIBRARY_PATH": "/tmp/malicious"},
        )
        assert result.exit_code != 0
        assert "blocked" in result.stderr.lower()

    async def test_node_options_blocked(self, dual_scheduler: Scheduler) -> None:
        """NODE_OPTIONS is blocked (Node.js runtime manipulation)."""
        result = await dual_scheduler.run(
            code="console.log('should not run')",
            language=Language.JAVASCRIPT,
            env_vars={"NODE_OPTIONS": "--expose-gc --max-old-space-size=8192"},
        )
        assert result.exit_code != 0
        assert "blocked" in result.stderr.lower()

    async def test_pythonstartup_blocked(self, dual_scheduler: Scheduler) -> None:
        """PYTHONSTARTUP is blocked (arbitrary code execution on Python start)."""
        result = await dual_scheduler.run(
            code="print('should not run')",
            language=Language.PYTHON,
            env_vars={"PYTHONSTARTUP": "/tmp/malicious.py"},
        )
        assert result.exit_code != 0
        assert "blocked" in result.stderr.lower()

    async def test_bash_env_blocked(self, dual_scheduler: Scheduler) -> None:
        """BASH_ENV is blocked (arbitrary code execution on bash start)."""
        result = await dual_scheduler.run(
            code="echo 'should not run'",
            language=Language.RAW,
            env_vars={"BASH_ENV": "/tmp/malicious.sh"},
        )
        assert result.exit_code != 0
        assert "blocked" in result.stderr.lower()

    async def test_path_blocked(self, dual_scheduler: Scheduler) -> None:
        """PATH is blocked (executable search path manipulation)."""
        result = await dual_scheduler.run(
            code="print('should not run')",
            language=Language.PYTHON,
            env_vars={"PATH": "/tmp/malicious:/usr/bin"},
        )
        assert result.exit_code != 0
        assert "blocked" in result.stderr.lower()

    async def test_glibc_tunables_blocked(self, dual_scheduler: Scheduler) -> None:
        """GLIBC_TUNABLES is blocked (CVE-2023-4911 mitigation)."""
        result = await dual_scheduler.run(
            code="print('should not run')",
            language=Language.PYTHON,
            env_vars={"GLIBC_TUNABLES": "glibc.tune.hwcaps=-AVX2"},
        )
        assert result.exit_code != 0
        assert "blocked" in result.stderr.lower()

    async def test_blocked_env_var_case_insensitive(self, dual_scheduler: Scheduler) -> None:
        """Blocked env var check is case-insensitive."""
        result = await dual_scheduler.run(
            code="print('should not run')",
            language=Language.PYTHON,
            env_vars={"ld_preload": "/tmp/malicious.so"},  # lowercase
        )
        assert result.exit_code != 0
        assert "blocked" in result.stderr.lower()

    # =========================================================================
    # Valid Environment Variables (Positive Tests)
    # =========================================================================

    async def test_custom_env_var_works(self, dual_scheduler: Scheduler) -> None:
        """Custom env vars are passed to executed code."""
        result = await dual_scheduler.run(
            code="import os; print(os.environ.get('MY_VAR', 'not set'))",
            language=Language.PYTHON,
            env_vars={"MY_VAR": "hello_world"},
        )
        assert result.exit_code == 0
        assert "hello_world" in result.stdout

    async def test_multiple_env_vars_work(self, dual_scheduler: Scheduler) -> None:
        """Multiple custom env vars work correctly."""
        result = await dual_scheduler.run(
            code="""
import os
print(os.environ.get('VAR1'))
print(os.environ.get('VAR2'))
print(os.environ.get('VAR3'))
""",
            language=Language.PYTHON,
            env_vars={"VAR1": "one", "VAR2": "two", "VAR3": "three"},
        )
        assert result.exit_code == 0
        assert "one" in result.stdout
        assert "two" in result.stdout
        assert "three" in result.stdout


class TestGuestAgentControlCharValidation:
    """Tests that Rust guest-agent rejects control characters (bypassing Pydantic).

    These tests use model_construct() to bypass Pydantic validation and send
    control characters directly to the VM, verifying the Rust defense-in-depth.
    """

    @pytest.fixture
    async def running_vm(self, vm_manager: VmManager) -> AsyncGenerator["QemuVM", None]:
        """Create a VM and yield it for testing."""
        from exec_sandbox.qemu_vm import QemuVM  # noqa: TC001

        vm: QemuVM = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="test-control-char",
            memory_mb=256,
            allow_network=False,
            allowed_domains=None,
        )
        try:
            yield vm
        finally:
            await vm_manager.destroy_vm(vm)

    async def _execute_with_bypass(
        self,
        vm: "QemuVM",
        env_vars: dict[str, str],
    ) -> tuple[int, str, str]:
        """Execute code bypassing Pydantic validation to test Rust validation.

        Uses model_construct() to create request without running validators.
        """
        from exec_sandbox import constants
        from exec_sandbox.guest_agent_protocol import (
            ExecutionCompleteMessage,
            OutputChunkMessage,
            StreamingErrorMessage,
        )

        # Bypass Pydantic validation using model_construct
        request = ExecuteCodeRequest.model_construct(
            action="exec",
            language=Language.PYTHON,
            code="print('should not run')",
            timeout=30,
            env_vars=env_vars,
        )

        # Connect and send directly to guest
        await vm.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        exit_code = -1

        async for msg in vm.channel.stream_messages(request, timeout=30):
            if isinstance(msg, OutputChunkMessage):
                if msg.type == "stdout":
                    stdout_chunks.append(msg.chunk)
                else:
                    stderr_chunks.append(msg.chunk)
            elif isinstance(msg, ExecutionCompleteMessage):
                exit_code = msg.exit_code
            elif isinstance(msg, StreamingErrorMessage):
                stderr_chunks.append(msg.message)
                break

        return exit_code, "".join(stdout_chunks), "".join(stderr_chunks)

    async def test_null_byte_rejected_by_rust(self, running_vm: "QemuVM") -> None:
        """Null byte in env var value is rejected by Rust guest-agent."""
        exit_code, _stdout, stderr = await self._execute_with_bypass(
            running_vm,
            env_vars={"FOO": "val\x00ue"},
        )
        assert exit_code != 0 or "control character" in stderr.lower()

    async def test_escape_sequence_rejected_by_rust(self, running_vm: "QemuVM") -> None:
        """ANSI escape sequence is rejected by Rust guest-agent."""
        exit_code, _stdout, stderr = await self._execute_with_bypass(
            running_vm,
            env_vars={"FOO": "\x1b[31mred\x1b[0m"},
        )
        assert exit_code != 0 or "control character" in stderr.lower()

    async def test_newline_rejected_by_rust(self, running_vm: "QemuVM") -> None:
        """Newline in env var value is rejected by Rust guest-agent."""
        exit_code, _stdout, stderr = await self._execute_with_bypass(
            running_vm,
            env_vars={"FOO": "line1\nline2"},
        )
        assert exit_code != 0 or "control character" in stderr.lower()

    async def test_carriage_return_rejected_by_rust(self, running_vm: "QemuVM") -> None:
        """Carriage return is rejected by Rust guest-agent."""
        exit_code, _stdout, stderr = await self._execute_with_bypass(
            running_vm,
            env_vars={"FOO": "start\roverwrite"},
        )
        assert exit_code != 0 or "control character" in stderr.lower()

    async def test_bell_rejected_by_rust(self, running_vm: "QemuVM") -> None:
        """Bell character (0x07) is rejected by Rust guest-agent."""
        exit_code, _stdout, stderr = await self._execute_with_bypass(
            running_vm,
            env_vars={"FOO": "ding\x07"},
        )
        assert exit_code != 0 or "control character" in stderr.lower()

    async def test_del_rejected_by_rust(self, running_vm: "QemuVM") -> None:
        """DEL character (0x7F) is rejected by Rust guest-agent."""
        exit_code, _stdout, stderr = await self._execute_with_bypass(
            running_vm,
            env_vars={"FOO": "delete\x7f"},
        )
        assert exit_code != 0 or "control character" in stderr.lower()

    async def test_tab_allowed_by_rust(self, running_vm: "QemuVM") -> None:
        """Tab character is allowed by Rust guest-agent."""
        # Use model_construct to bypass Pydantic, proving Rust allows tabs
        request = ExecuteCodeRequest.model_construct(
            action="exec",
            language=Language.PYTHON,
            code="import os; print(repr(os.environ.get('FOO')))",
            timeout=30,
            env_vars={"FOO": "col1\tcol2"},
        )

        from exec_sandbox import constants
        from exec_sandbox.guest_agent_protocol import (
            ExecutionCompleteMessage,
            OutputChunkMessage,
        )

        await running_vm.channel.connect(constants.GUEST_CONNECT_TIMEOUT_SECONDS)

        stdout = ""
        exit_code = -1
        async for msg in running_vm.channel.stream_messages(request, timeout=30):
            if isinstance(msg, OutputChunkMessage) and msg.type == "stdout":
                stdout += msg.chunk
            elif isinstance(msg, ExecutionCompleteMessage):
                exit_code = msg.exit_code

        assert exit_code == 0
        assert "col1\\tcol2" in stdout

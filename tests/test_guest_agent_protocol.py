"""Unit tests for guest agent protocol models.

Tests Pydantic serialization, validation, and discriminated unions.
No mocks - pure model testing.
"""

import pytest
from pydantic import ValidationError

from exec_sandbox.guest_agent_protocol import (
    ExecuteCodeRequest,
    ExecutionCompleteMessage,
    InstallPackagesRequest,
    OutputChunkMessage,
    PingRequest,
    PongMessage,
    StreamingErrorMessage,
    StreamingMessage,
)
from exec_sandbox.models import Language

# ============================================================================
# Request Models
# ============================================================================


class TestPingRequest:
    """Tests for PingRequest model."""

    def test_default_action(self) -> None:
        """PingRequest has default action='ping'."""
        req = PingRequest()
        assert req.action == "ping"

    def test_serialize_to_dict(self) -> None:
        """PingRequest serializes correctly."""
        req = PingRequest()
        data = req.model_dump()
        assert data == {"action": "ping"}

    def test_serialize_to_json(self) -> None:
        """PingRequest serializes to JSON."""
        req = PingRequest()
        json_str = req.model_dump_json()
        assert json_str == '{"action":"ping"}'


class TestExecuteCodeRequest:
    """Tests for ExecuteCodeRequest model."""

    def test_minimal_request(self) -> None:
        """ExecuteCodeRequest with required fields only."""
        req = ExecuteCodeRequest(
            language=Language.PYTHON,
            code="print('hello')",
        )
        assert req.action == "exec"
        assert req.language == "python"
        assert req.code == "print('hello')"
        assert req.timeout == 0  # default
        assert req.env_vars == {}  # default

    def test_full_request(self) -> None:
        """ExecuteCodeRequest with all fields."""
        req = ExecuteCodeRequest(
            language=Language.JAVASCRIPT,
            code="console.log('hello')",
            timeout=60,
            env_vars={"FOO": "bar", "BAZ": "qux"},
        )
        assert req.language == "javascript"
        assert req.timeout == 60
        assert req.env_vars == {"FOO": "bar", "BAZ": "qux"}

    def test_language_validation(self) -> None:
        """ExecuteCodeRequest rejects invalid languages."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(language="ruby", code="puts 'hello'")
        assert "language" in str(exc_info.value)

    def test_timeout_range(self) -> None:
        """ExecuteCodeRequest enforces timeout range 0-300."""
        # Valid: 0
        req = ExecuteCodeRequest(language=Language.PYTHON, code="x", timeout=0)
        assert req.timeout == 0

        # Valid: 300
        req = ExecuteCodeRequest(language=Language.PYTHON, code="x", timeout=300)
        assert req.timeout == 300

        # Invalid: negative
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(language=Language.PYTHON, code="x", timeout=-1)

        # Invalid: > 300
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(language=Language.PYTHON, code="x", timeout=301)

    def test_code_max_length(self) -> None:
        """ExecuteCodeRequest enforces 1MB code limit."""
        # Valid: 1MB exactly
        large_code = "x" * 1_000_000
        req = ExecuteCodeRequest(language=Language.PYTHON, code=large_code)
        assert len(req.code) == 1_000_000

        # Invalid: > 1MB
        too_large = "x" * 1_000_001
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(language=Language.PYTHON, code=too_large)

    def test_serialize_json(self) -> None:
        """ExecuteCodeRequest serializes to JSON correctly."""
        req = ExecuteCodeRequest(
            language=Language.PYTHON,
            code="print(1)",
            timeout=30,
            env_vars={"KEY": "value"},
        )
        data = req.model_dump()
        assert data["action"] == "exec"
        assert data["language"] == "python"
        assert data["code"] == "print(1)"
        assert data["timeout"] == 30
        assert data["env_vars"] == {"KEY": "value"}


class TestInstallPackagesRequest:
    """Tests for InstallPackagesRequest model."""

    def test_minimal_request(self) -> None:
        """InstallPackagesRequest with required fields."""
        req = InstallPackagesRequest(
            language=Language.PYTHON,
            packages=["pandas==2.0.0"],
        )
        assert req.action == "install_packages"
        assert req.language == "python"
        assert req.packages == ["pandas==2.0.0"]
        assert req.timeout == 300  # default

    def test_multiple_packages(self) -> None:
        """InstallPackagesRequest with multiple packages."""
        req = InstallPackagesRequest(
            language=Language.JAVASCRIPT,
            packages=["lodash@4.17.21", "axios@1.6.0", "react@18.2.0"],
            timeout=120,
        )
        assert len(req.packages) == 3
        assert req.timeout == 120

    def test_packages_min_length(self) -> None:
        """InstallPackagesRequest requires at least 1 package."""
        with pytest.raises(ValidationError):
            InstallPackagesRequest(language=Language.PYTHON, packages=[])

    def test_packages_max_length(self) -> None:
        """InstallPackagesRequest allows max 50 packages."""
        # Valid: 50 packages
        packages = [f"pkg{i}==1.0.0" for i in range(50)]
        req = InstallPackagesRequest(language=Language.PYTHON, packages=packages)
        assert len(req.packages) == 50

        # Invalid: 51 packages
        packages = [f"pkg{i}==1.0.0" for i in range(51)]
        with pytest.raises(ValidationError):
            InstallPackagesRequest(language=Language.PYTHON, packages=packages)

    def test_timeout_range(self) -> None:
        """InstallPackagesRequest enforces timeout range 0-300."""
        # Valid: max
        req = InstallPackagesRequest(language=Language.PYTHON, packages=["x==1.0"], timeout=300)
        assert req.timeout == 300

        # Invalid: > 300
        with pytest.raises(ValidationError):
            InstallPackagesRequest(language=Language.PYTHON, packages=["x==1.0"], timeout=301)


# ============================================================================
# Response Models
# ============================================================================


class TestOutputChunkMessage:
    """Tests for OutputChunkMessage model."""

    def test_stdout_chunk(self) -> None:
        """OutputChunkMessage for stdout."""
        msg = OutputChunkMessage(type="stdout", chunk="Hello, World!")
        assert msg.type == "stdout"
        assert msg.chunk == "Hello, World!"

    def test_stderr_chunk(self) -> None:
        """OutputChunkMessage for stderr."""
        msg = OutputChunkMessage(type="stderr", chunk="Error: something failed")
        assert msg.type == "stderr"
        assert msg.chunk == "Error: something failed"

    def test_invalid_type(self) -> None:
        """OutputChunkMessage rejects invalid types."""
        with pytest.raises(ValidationError):
            OutputChunkMessage(type="stdin", chunk="data")

    def test_chunk_max_length(self) -> None:
        """OutputChunkMessage enforces 10MB chunk limit."""
        # Valid: 1MB (well under 10MB limit)
        chunk = "x" * 1_000_000
        msg = OutputChunkMessage(type="stdout", chunk=chunk)
        assert len(msg.chunk) == 1_000_000

        # Invalid: > 10MB
        too_large = "x" * 10_000_001
        with pytest.raises(ValidationError):
            OutputChunkMessage(type="stdout", chunk=too_large)

    def test_empty_chunk(self) -> None:
        """OutputChunkMessage allows empty chunk."""
        msg = OutputChunkMessage(type="stdout", chunk="")
        assert msg.chunk == ""


class TestExecutionCompleteMessage:
    """Tests for ExecutionCompleteMessage model."""

    def test_success(self) -> None:
        """ExecutionCompleteMessage for successful execution."""
        msg = ExecutionCompleteMessage(exit_code=0, execution_time_ms=150)
        assert msg.type == "complete"
        assert msg.exit_code == 0
        assert msg.execution_time_ms == 150

    def test_failure(self) -> None:
        """ExecutionCompleteMessage for failed execution."""
        msg = ExecutionCompleteMessage(exit_code=1, execution_time_ms=50)
        assert msg.exit_code == 1

    def test_negative_exit_code(self) -> None:
        """ExecutionCompleteMessage allows negative exit codes (signals)."""
        msg = ExecutionCompleteMessage(exit_code=-9, execution_time_ms=100)
        assert msg.exit_code == -9


class TestPongMessage:
    """Tests for PongMessage model."""

    def test_pong(self) -> None:
        """PongMessage with version."""
        msg = PongMessage(version="1.0.0")
        assert msg.type == "pong"
        assert msg.version == "1.0.0"

    def test_serialize(self) -> None:
        """PongMessage serializes correctly."""
        msg = PongMessage(version="2.1.0")
        data = msg.model_dump()
        assert data == {"type": "pong", "version": "2.1.0"}


class TestStreamingErrorMessage:
    """Tests for StreamingErrorMessage model."""

    def test_error_without_version(self) -> None:
        """StreamingErrorMessage without version."""
        msg = StreamingErrorMessage(
            message="Timeout exceeded",
            error_type="timeout",
        )
        assert msg.type == "error"
        assert msg.message == "Timeout exceeded"
        assert msg.error_type == "timeout"
        assert msg.version is None

    def test_error_with_version(self) -> None:
        """StreamingErrorMessage with version."""
        msg = StreamingErrorMessage(
            message="Internal error",
            error_type="internal",
            version="1.0.0",
        )
        assert msg.version == "1.0.0"


# ============================================================================
# Discriminated Union
# ============================================================================


class TestStreamingMessage:
    """Tests for StreamingMessage discriminated union."""

    def test_parse_stdout_chunk(self) -> None:
        """Parse stdout OutputChunkMessage from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {"type": "stdout", "chunk": "Hello"}
        msg = adapter.validate_python(data)
        assert isinstance(msg, OutputChunkMessage)
        assert msg.type == "stdout"
        assert msg.chunk == "Hello"

    def test_parse_stderr_chunk(self) -> None:
        """Parse stderr OutputChunkMessage from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {"type": "stderr", "chunk": "Error!"}
        msg = adapter.validate_python(data)
        assert isinstance(msg, OutputChunkMessage)
        assert msg.type == "stderr"

    def test_parse_complete(self) -> None:
        """Parse ExecutionCompleteMessage from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {"type": "complete", "exit_code": 0, "execution_time_ms": 100}
        msg = adapter.validate_python(data)
        assert isinstance(msg, ExecutionCompleteMessage)
        assert msg.exit_code == 0

    def test_parse_pong(self) -> None:
        """Parse PongMessage from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {"type": "pong", "version": "1.0.0"}
        msg = adapter.validate_python(data)
        assert isinstance(msg, PongMessage)
        assert msg.version == "1.0.0"

    def test_parse_error(self) -> None:
        """Parse StreamingErrorMessage from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {"type": "error", "message": "Failed", "error_type": "timeout"}
        msg = adapter.validate_python(data)
        assert isinstance(msg, StreamingErrorMessage)
        assert msg.message == "Failed"

    def test_parse_unknown_type(self) -> None:
        """Reject unknown message type."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {"type": "unknown", "data": "something"}
        with pytest.raises(ValidationError):
            adapter.validate_python(data)

    def test_parse_from_json(self) -> None:
        """Parse StreamingMessage from JSON string."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        json_str = '{"type": "complete", "exit_code": 0, "execution_time_ms": 42}'
        msg = adapter.validate_json(json_str)
        assert isinstance(msg, ExecutionCompleteMessage)
        assert msg.execution_time_ms == 42

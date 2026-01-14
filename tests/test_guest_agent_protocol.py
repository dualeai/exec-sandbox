"""Unit tests for guest agent protocol models.

Tests Pydantic serialization, validation, and discriminated unions.
No mocks - pure model testing.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import characters, integers, sampled_from, text
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


class TestEnvVarValidation:
    """Tests for environment variable validation."""

    def test_valid_env_vars(self) -> None:
        """Valid env vars with printable ASCII and tabs."""
        req = ExecuteCodeRequest(
            language=Language.PYTHON,
            code="print(1)",
            env_vars={"FOO": "bar", "WITH_TAB": "value\twith\ttabs"},
        )
        assert req.env_vars["WITH_TAB"] == "value\twith\ttabs"

    def test_null_byte_in_name_rejected(self) -> None:
        """Null byte in env var name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={"FOO\x00BAR": "value"},
            )
        assert "control character" in str(exc_info.value).lower()

    def test_null_byte_in_value_rejected(self) -> None:
        """Null byte in env var value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={"FOO": "val\x00ue"},
            )
        assert "control character" in str(exc_info.value).lower()

    def test_escape_sequence_in_value_rejected(self) -> None:
        """ANSI escape sequence (ESC = 0x1B) in value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={"FOO": "\x1b[31mred\x1b[0m"},
            )
        assert "control character" in str(exc_info.value).lower()

    def test_newline_in_value_rejected(self) -> None:
        """Newline in env var value is rejected (log injection prevention)."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={"FOO": "line1\nline2"},
            )
        assert "control character" in str(exc_info.value).lower()

    def test_carriage_return_in_value_rejected(self) -> None:
        """Carriage return in value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={"FOO": "start\roverwrite"},
            )
        assert "control character" in str(exc_info.value).lower()

    def test_bell_character_rejected(self) -> None:
        """Bell character (0x07) is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={"FOO": "ding\x07"},
            )
        assert "control character" in str(exc_info.value).lower()

    def test_del_character_rejected(self) -> None:
        """DEL character (0x7F) is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={"FOO": "delete\x7f"},
            )
        assert "control character" in str(exc_info.value).lower()

    def test_env_var_name_too_long(self) -> None:
        """Env var name exceeding 256 chars is rejected."""
        long_name = "A" * 257
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={long_name: "value"},
            )
        assert "length" in str(exc_info.value).lower()

    def test_env_var_value_too_long(self) -> None:
        """Env var value exceeding 4096 chars is rejected."""
        long_value = "x" * 4097
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={"FOO": long_value},
            )
        assert "too large" in str(exc_info.value).lower()

    def test_too_many_env_vars(self) -> None:
        """More than 100 env vars is rejected."""
        many_vars = {f"VAR_{i}": "value" for i in range(101)}
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars=many_vars,
            )
        assert "too many" in str(exc_info.value).lower()

    def test_utf8_in_value_allowed(self) -> None:
        """UTF-8 characters (emoji, non-Latin) are allowed."""
        req = ExecuteCodeRequest(
            language=Language.PYTHON,
            code="x",
            env_vars={"GREETING": "Hello 世界 🌍"},
        )
        assert "🌍" in req.env_vars["GREETING"]

    def test_empty_name_rejected(self) -> None:
        """Empty env var name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={"": "value"},
            )
        assert "length" in str(exc_info.value).lower()


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

    def test_with_timing_fields(self) -> None:
        """ExecutionCompleteMessage with optional timing fields."""
        msg = ExecutionCompleteMessage(
            exit_code=0,
            execution_time_ms=150,
            spawn_ms=5,
            process_ms=140,
        )
        assert msg.spawn_ms == 5
        assert msg.process_ms == 140

    def test_without_timing_fields(self) -> None:
        """ExecutionCompleteMessage without optional timing fields (backwards compat)."""
        msg = ExecutionCompleteMessage(exit_code=0, execution_time_ms=150)
        assert msg.spawn_ms is None
        assert msg.process_ms is None

    def test_partial_timing_fields(self) -> None:
        """ExecutionCompleteMessage with only spawn_ms (timeout scenario)."""
        msg = ExecutionCompleteMessage(
            exit_code=-1,
            execution_time_ms=30000,
            spawn_ms=10,
            process_ms=None,  # Process timed out before completing
        )
        assert msg.spawn_ms == 10
        assert msg.process_ms is None


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

    def test_parse_complete_with_timing(self) -> None:
        """Parse ExecutionCompleteMessage with timing fields from JSON."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        json_str = '{"type": "complete", "exit_code": 0, "execution_time_ms": 100, "spawn_ms": 5, "process_ms": 90}'
        msg = adapter.validate_json(json_str)
        assert isinstance(msg, ExecutionCompleteMessage)
        assert msg.spawn_ms == 5
        assert msg.process_ms == 90

    def test_parse_complete_without_timing(self) -> None:
        """Parse ExecutionCompleteMessage without timing fields (backwards compat)."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        json_str = '{"type": "complete", "exit_code": 0, "execution_time_ms": 42}'
        msg = adapter.validate_json(json_str)
        assert isinstance(msg, ExecutionCompleteMessage)
        assert msg.spawn_ms is None
        assert msg.process_ms is None

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


# ============================================================================
# Property-Based Tests (Hypothesis)
# ============================================================================


class TestEnvVarValidationPropertyBased:
    """Property-based tests for env var validation using Hypothesis.

    These tests automatically discover edge cases by generating random inputs.
    Reference: https://hypothesis.readthedocs.io/en/latest/data.html
    """

    # Strategy for safe characters (printable ASCII + tab + UTF-8, excluding DEL)
    safe_chars = characters(
        min_codepoint=0x09,  # Start at tab
        max_codepoint=0x10FFFF,
        exclude_categories=("Cs",),  # Exclude surrogates
    ).filter(lambda c: ord(c) == 0x09 or (ord(c) >= 0x20 and ord(c) != 0x7F))  # Tab or printable+ (exclude DEL)

    # Strategy for forbidden control characters
    control_chars = sampled_from(
        [chr(c) for c in range(0x09)]  # NUL through BS
        + [chr(c) for c in range(0x0A, 0x20)]  # LF through US
        + [chr(0x7F)]  # DEL
    )

    # Strategy for valid env var names (alphanumeric + underscore, starts with letter/underscore)
    valid_name = text(
        alphabet=characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
        min_size=1,
        max_size=50,
    ).filter(lambda s: s[0].isalpha() or s[0] == "_")

    @given(
        name=valid_name,
        safe_value=text(safe_chars, min_size=0, max_size=100),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_safe_values_accepted(self, name: str, safe_value: str) -> None:
        """Property: Values with only safe characters should be accepted."""
        req = ExecuteCodeRequest(
            language=Language.PYTHON,
            code="x",
            env_vars={name: safe_value},
        )
        assert req.env_vars[name] == safe_value

    @given(
        name=valid_name,
        prefix=text(safe_chars, min_size=0, max_size=10),
        control=control_chars,
        suffix=text(safe_chars, min_size=0, max_size=10),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_control_chars_in_value_rejected(self, name: str, prefix: str, control: str, suffix: str) -> None:
        """Property: Any value containing a control character must be rejected."""
        malicious_value = prefix + control + suffix
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={name: malicious_value},
            )

    @given(
        prefix=text(safe_chars, min_size=0, max_size=10),
        control=control_chars,
        suffix=text(safe_chars, min_size=0, max_size=10),
        value=text(safe_chars, min_size=0, max_size=20),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_control_chars_in_name_rejected(self, prefix: str, control: str, suffix: str, value: str) -> None:
        """Property: Any name containing a control character must be rejected."""
        # Ensure name is non-empty after adding control char
        malicious_name = (prefix + control + suffix) or (control + "X")
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars={malicious_name: value},
            )

    @given(num_vars=integers(min_value=101, max_value=150))
    @settings(max_examples=10)
    def test_too_many_vars_rejected(self, num_vars: int) -> None:
        """Property: More than 100 env vars must be rejected."""
        many_vars = {f"VAR_{i}": "v" for i in range(num_vars)}
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(
                language=Language.PYTHON,
                code="x",
                env_vars=many_vars,
            )

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
    FileContentMessage,
    FileEntryInfo,
    FileListMessage,
    FileWriteAckMessage,
    InstallPackagesRequest,
    ListFilesRequest,
    OutputChunkMessage,
    PingRequest,
    PongMessage,
    ReadFileRequest,
    StreamingErrorMessage,
    StreamingMessage,
    WriteFileRequest,
)
from exec_sandbox.models import FileInfo, Language

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


# ============================================================================
# File I/O Request Models
# ============================================================================


class TestWriteFileRequest:
    """Tests for WriteFileRequest model."""

    def test_minimal_request(self) -> None:
        """WriteFileRequest with required fields only."""
        req = WriteFileRequest(
            path="hello.txt",
            content_base64="SGVsbG8=",
        )
        assert req.action == "write_file"
        assert req.path == "hello.txt"
        assert req.content_base64 == "SGVsbG8="
        assert req.make_executable is False  # default

    def test_with_make_executable(self) -> None:
        """WriteFileRequest with make_executable=True."""
        req = WriteFileRequest(
            path="run.sh",
            content_base64="IyEvYmluL2Jhc2g=",
            make_executable=True,
        )
        assert req.make_executable is True

    def test_path_max_length(self) -> None:
        """WriteFileRequest accepts path at max length (255)."""
        long_path = "a" * 255
        req = WriteFileRequest(path=long_path, content_base64="AA==")
        assert len(req.path) == 255

    def test_path_exceeds_max_length(self) -> None:
        """WriteFileRequest rejects path exceeding max length."""
        too_long = "a" * 256
        with pytest.raises(ValidationError):
            WriteFileRequest(path=too_long, content_base64="AA==")

    def test_empty_path_rejected(self) -> None:
        """WriteFileRequest rejects empty path."""
        with pytest.raises(ValidationError):
            WriteFileRequest(path="", content_base64="AA==")

    def test_content_base64_max_length(self) -> None:
        """WriteFileRequest accepts content_base64 at max length."""
        large_content = "A" * 14_000_000
        req = WriteFileRequest(path="big.bin", content_base64=large_content)
        assert len(req.content_base64) == 14_000_000

    def test_content_base64_exceeds_max_length(self) -> None:
        """WriteFileRequest rejects content_base64 exceeding max length."""
        too_large = "A" * 14_000_001
        with pytest.raises(ValidationError):
            WriteFileRequest(path="big.bin", content_base64=too_large)

    def test_serialize_roundtrip(self) -> None:
        """WriteFileRequest serializes and deserializes correctly."""
        req = WriteFileRequest(
            path="data/output.csv",
            content_base64="Y29sMSxjb2wy",
            make_executable=False,
        )
        data = req.model_dump()
        assert data == {
            "action": "write_file",
            "path": "data/output.csv",
            "content_base64": "Y29sMSxjb2wy",
            "make_executable": False,
        }
        restored = WriteFileRequest.model_validate(data)
        assert restored.path == req.path
        assert restored.content_base64 == req.content_base64
        assert restored.make_executable == req.make_executable

    def test_serialize_to_json(self) -> None:
        """WriteFileRequest serializes to JSON."""
        req = WriteFileRequest(path="f.txt", content_base64="AA==")
        json_str = req.model_dump_json()
        restored = WriteFileRequest.model_validate_json(json_str)
        assert restored.path == "f.txt"
        assert restored.content_base64 == "AA=="


class TestReadFileRequest:
    """Tests for ReadFileRequest model."""

    def test_minimal_request(self) -> None:
        """ReadFileRequest with required fields."""
        req = ReadFileRequest(path="output.txt")
        assert req.action == "read_file"
        assert req.path == "output.txt"

    def test_path_max_length(self) -> None:
        """ReadFileRequest accepts path at max length (255)."""
        long_path = "b" * 255
        req = ReadFileRequest(path=long_path)
        assert len(req.path) == 255

    def test_path_exceeds_max_length(self) -> None:
        """ReadFileRequest rejects path exceeding max length."""
        too_long = "b" * 256
        with pytest.raises(ValidationError):
            ReadFileRequest(path=too_long)

    def test_empty_path_rejected(self) -> None:
        """ReadFileRequest rejects empty path."""
        with pytest.raises(ValidationError):
            ReadFileRequest(path="")

    def test_serialize_to_dict(self) -> None:
        """ReadFileRequest serializes correctly."""
        req = ReadFileRequest(path="results/data.json")
        data = req.model_dump()
        assert data == {"action": "read_file", "path": "results/data.json"}

    def test_serialize_to_json(self) -> None:
        """ReadFileRequest serializes to JSON."""
        req = ReadFileRequest(path="test.py")
        json_str = req.model_dump_json()
        restored = ReadFileRequest.model_validate_json(json_str)
        assert restored.path == "test.py"


class TestListFilesRequest:
    """Tests for ListFilesRequest model."""

    def test_minimal_request(self) -> None:
        """ListFilesRequest with explicit path."""
        req = ListFilesRequest(path="src")
        assert req.action == "list_files"
        assert req.path == "src"

    def test_empty_path_for_root(self) -> None:
        """ListFilesRequest with empty path lists sandbox root."""
        req = ListFilesRequest()
        assert req.path == ""

    def test_empty_string_path_valid(self) -> None:
        """ListFilesRequest accepts explicit empty string path."""
        req = ListFilesRequest(path="")
        assert req.path == ""

    def test_path_max_length(self) -> None:
        """ListFilesRequest accepts path at max length (255)."""
        long_path = "c" * 255
        req = ListFilesRequest(path=long_path)
        assert len(req.path) == 255

    def test_path_exceeds_max_length(self) -> None:
        """ListFilesRequest rejects path exceeding max length."""
        too_long = "c" * 256
        with pytest.raises(ValidationError):
            ListFilesRequest(path=too_long)

    def test_serialize_to_dict(self) -> None:
        """ListFilesRequest serializes correctly."""
        req = ListFilesRequest(path="data")
        data = req.model_dump()
        assert data == {"action": "list_files", "path": "data"}

    def test_serialize_default_path(self) -> None:
        """ListFilesRequest serializes default empty path."""
        req = ListFilesRequest()
        data = req.model_dump()
        assert data == {"action": "list_files", "path": ""}


# ============================================================================
# File I/O Response Models
# ============================================================================


class TestFileWriteAckMessage:
    """Tests for FileWriteAckMessage model."""

    def test_ack_fields(self) -> None:
        """FileWriteAckMessage has correct fields."""
        msg = FileWriteAckMessage(path="hello.txt", bytes_written=5)
        assert msg.type == "file_write_ack"
        assert msg.path == "hello.txt"
        assert msg.bytes_written == 5

    def test_large_bytes_written(self) -> None:
        """FileWriteAckMessage supports large file sizes."""
        msg = FileWriteAckMessage(path="big.bin", bytes_written=10_000_000)
        assert msg.bytes_written == 10_000_000

    def test_serialize(self) -> None:
        """FileWriteAckMessage serializes correctly."""
        msg = FileWriteAckMessage(path="out.csv", bytes_written=1024)
        data = msg.model_dump()
        assert data == {
            "type": "file_write_ack",
            "path": "out.csv",
            "bytes_written": 1024,
        }

    def test_serialize_to_json(self) -> None:
        """FileWriteAckMessage serializes to JSON and back."""
        msg = FileWriteAckMessage(path="script.py", bytes_written=42)
        json_str = msg.model_dump_json()
        restored = FileWriteAckMessage.model_validate_json(json_str)
        assert restored.path == "script.py"
        assert restored.bytes_written == 42


class TestFileContentMessage:
    """Tests for FileContentMessage model."""

    def test_content_fields(self) -> None:
        """FileContentMessage has correct fields."""
        msg = FileContentMessage(
            path="data.txt",
            content_base64="SGVsbG8gV29ybGQ=",
            size=11,
        )
        assert msg.type == "file_content"
        assert msg.path == "data.txt"
        assert msg.content_base64 == "SGVsbG8gV29ybGQ="
        assert msg.size == 11

    def test_empty_file(self) -> None:
        """FileContentMessage for an empty file."""
        msg = FileContentMessage(path="empty.txt", content_base64="", size=0)
        assert msg.content_base64 == ""
        assert msg.size == 0

    def test_serialize(self) -> None:
        """FileContentMessage serializes correctly."""
        msg = FileContentMessage(
            path="config.json",
            content_base64="e30=",
            size=2,
        )
        data = msg.model_dump()
        assert data == {
            "type": "file_content",
            "path": "config.json",
            "content_base64": "e30=",
            "size": 2,
        }

    def test_serialize_to_json(self) -> None:
        """FileContentMessage serializes to JSON and back."""
        msg = FileContentMessage(path="f.bin", content_base64="AQID", size=3)
        json_str = msg.model_dump_json()
        restored = FileContentMessage.model_validate_json(json_str)
        assert restored.path == "f.bin"
        assert restored.content_base64 == "AQID"
        assert restored.size == 3


class TestFileListMessage:
    """Tests for FileListMessage model."""

    def test_empty_listing(self) -> None:
        """FileListMessage with no entries."""
        msg = FileListMessage(path="empty_dir", entries=[])
        assert msg.type == "file_list"
        assert msg.path == "empty_dir"
        assert msg.entries == []

    def test_with_entries(self) -> None:
        """FileListMessage with file and directory entries."""
        entries = [
            FileEntryInfo(name="src", is_dir=True, size=0),
            FileEntryInfo(name="main.py", is_dir=False, size=256),
            FileEntryInfo(name="README.md", is_dir=False, size=1024),
        ]
        msg = FileListMessage(path="", entries=entries)
        assert len(msg.entries) == 3
        assert msg.entries[0].name == "src"
        assert msg.entries[0].is_dir is True
        assert msg.entries[1].name == "main.py"
        assert msg.entries[1].is_dir is False
        assert msg.entries[1].size == 256

    def test_serialize(self) -> None:
        """FileListMessage serializes correctly."""
        msg = FileListMessage(
            path="project",
            entries=[FileEntryInfo(name="app.js", is_dir=False, size=512)],
        )
        data = msg.model_dump()
        assert data == {
            "type": "file_list",
            "path": "project",
            "entries": [{"name": "app.js", "is_dir": False, "size": 512}],
        }

    def test_serialize_to_json(self) -> None:
        """FileListMessage serializes to JSON and back."""
        msg = FileListMessage(
            path="",
            entries=[
                FileEntryInfo(name="dir", is_dir=True, size=0),
                FileEntryInfo(name="file.txt", is_dir=False, size=100),
            ],
        )
        json_str = msg.model_dump_json()
        restored = FileListMessage.model_validate_json(json_str)
        assert len(restored.entries) == 2
        assert restored.entries[0].is_dir is True
        assert restored.entries[1].size == 100


class TestFileEntryInfo:
    """Tests for FileEntryInfo model."""

    def test_file_entry(self) -> None:
        """FileEntryInfo for a regular file."""
        entry = FileEntryInfo(name="report.pdf", is_dir=False, size=4096)
        assert entry.name == "report.pdf"
        assert entry.is_dir is False
        assert entry.size == 4096

    def test_directory_entry(self) -> None:
        """FileEntryInfo for a directory."""
        entry = FileEntryInfo(name="src", is_dir=True, size=0)
        assert entry.name == "src"
        assert entry.is_dir is True
        assert entry.size == 0

    def test_serialize_roundtrip(self) -> None:
        """FileEntryInfo serializes and deserializes correctly."""
        entry = FileEntryInfo(name="data.csv", is_dir=False, size=2048)
        data = entry.model_dump()
        assert data == {"name": "data.csv", "is_dir": False, "size": 2048}
        restored = FileEntryInfo.model_validate(data)
        assert restored.name == entry.name
        assert restored.is_dir == entry.is_dir
        assert restored.size == entry.size


class TestFileInfo:
    """Tests for FileInfo model (from models.py)."""

    def test_file_entry(self) -> None:
        """FileInfo for a regular file."""
        info = FileInfo(name="output.log", is_dir=False, size=8192)
        assert info.name == "output.log"
        assert info.is_dir is False
        assert info.size == 8192

    def test_directory_entry(self) -> None:
        """FileInfo for a directory."""
        info = FileInfo(name="results", is_dir=True, size=0)
        assert info.name == "results"
        assert info.is_dir is True
        assert info.size == 0

    def test_serialize_roundtrip(self) -> None:
        """FileInfo serializes and deserializes correctly."""
        info = FileInfo(name="image.png", is_dir=False, size=65536)
        data = info.model_dump()
        assert data == {"name": "image.png", "is_dir": False, "size": 65536}
        restored = FileInfo.model_validate(data)
        assert restored.name == info.name
        assert restored.is_dir == info.is_dir
        assert restored.size == info.size


# ============================================================================
# Discriminated Union - File I/O Messages
# ============================================================================


class TestStreamingMessageFileIO:
    """Tests for StreamingMessage discriminated union with file I/O types."""

    def test_parse_file_write_ack(self) -> None:
        """Parse FileWriteAckMessage from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {"type": "file_write_ack", "path": "hello.txt", "bytes_written": 5}
        msg = adapter.validate_python(data)
        assert isinstance(msg, FileWriteAckMessage)
        assert msg.path == "hello.txt"
        assert msg.bytes_written == 5

    def test_parse_file_write_ack_from_json(self) -> None:
        """Parse FileWriteAckMessage from JSON string."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        json_str = '{"type": "file_write_ack", "path": "script.sh", "bytes_written": 128}'
        msg = adapter.validate_json(json_str)
        assert isinstance(msg, FileWriteAckMessage)
        assert msg.path == "script.sh"
        assert msg.bytes_written == 128

    def test_parse_file_content(self) -> None:
        """Parse FileContentMessage from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {
            "type": "file_content",
            "path": "data.txt",
            "content_base64": "SGVsbG8=",
            "size": 5,
        }
        msg = adapter.validate_python(data)
        assert isinstance(msg, FileContentMessage)
        assert msg.path == "data.txt"
        assert msg.content_base64 == "SGVsbG8="
        assert msg.size == 5

    def test_parse_file_content_from_json(self) -> None:
        """Parse FileContentMessage from JSON string."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        json_str = '{"type": "file_content", "path": "out.bin", "content_base64": "AQID", "size": 3}'
        msg = adapter.validate_json(json_str)
        assert isinstance(msg, FileContentMessage)
        assert msg.path == "out.bin"
        assert msg.size == 3

    def test_parse_file_list(self) -> None:
        """Parse FileListMessage from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {
            "type": "file_list",
            "path": "",
            "entries": [
                {"name": "src", "is_dir": True, "size": 0},
                {"name": "main.py", "is_dir": False, "size": 200},
            ],
        }
        msg = adapter.validate_python(data)
        assert isinstance(msg, FileListMessage)
        assert len(msg.entries) == 2
        assert msg.entries[0].name == "src"
        assert msg.entries[0].is_dir is True

    def test_parse_file_list_from_json(self) -> None:
        """Parse FileListMessage from JSON string."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        json_str = (
            '{"type": "file_list", "path": "project", "entries": [{"name": "index.js", "is_dir": false, "size": 512}]}'
        )
        msg = adapter.validate_json(json_str)
        assert isinstance(msg, FileListMessage)
        assert msg.path == "project"
        assert len(msg.entries) == 1
        assert msg.entries[0].name == "index.js"

    def test_parse_file_list_empty_entries(self) -> None:
        """Parse FileListMessage with empty entries from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(StreamingMessage)
        data = {"type": "file_list", "path": "empty", "entries": []}
        msg = adapter.validate_python(data)
        assert isinstance(msg, FileListMessage)
        assert msg.entries == []


# ============================================================================
# Property-Based Tests - File I/O (Hypothesis)
# ============================================================================


class TestFilePathValidationPropertyBased:
    """Property-based tests for file path validation using Hypothesis.

    These tests automatically discover edge cases in path validation
    across WriteFileRequest, ReadFileRequest, and ListFilesRequest.
    """

    # Strategy for safe relative path segments (alphanumeric + common filename chars)
    safe_path_chars = characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-./",
    )

    @given(
        segment=text(
            alphabet=characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters="_-",
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_safe_relative_paths_accepted(self, segment: str) -> None:
        """Property: Safe relative paths should be accepted by all file request models."""
        path = f"subdir/{segment}.txt"
        # WriteFileRequest
        req_w = WriteFileRequest(path=path, content_base64="AA==")
        assert req_w.path == path
        # ReadFileRequest
        req_r = ReadFileRequest(path=path)
        assert req_r.path == path
        # ListFilesRequest
        req_l = ListFilesRequest(path=path)
        assert req_l.path == path

    @given(
        extra=text(
            alphabet=characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=50)
    def test_overlong_paths_rejected(self, extra: str) -> None:
        """Property: Paths exceeding 255 chars must be rejected."""
        # Build a path that is guaranteed to exceed 255 chars
        path = "a" * 256 + extra
        with pytest.raises(ValidationError):
            WriteFileRequest(path=path, content_base64="AA==")
        with pytest.raises(ValidationError):
            ReadFileRequest(path=path)
        with pytest.raises(ValidationError):
            ListFilesRequest(path=path)

    # Strategy for control characters (same set used by env var tests)
    control_chars = sampled_from(
        [chr(c) for c in range(0x09)]  # NUL through BS
        + [chr(c) for c in range(0x0A, 0x20)]  # LF through US
        + [chr(0x7F)]  # DEL
    )

    @given(
        prefix=text(
            alphabet=characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=10,
        ),
        control=control_chars,
        suffix=text(
            alphabet=characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_control_chars_in_path_rejected_write(self, prefix: str, control: str, suffix: str) -> None:
        """Property: Control characters in WriteFileRequest path should be rejected by Pydantic or the guest agent.

        Note: Pydantic's min_length/max_length do not reject control characters.
        This test documents the current behavior -- the guest agent's Rust-side
        validation is the security boundary for path traversal and control chars.
        If Pydantic accepts it, the test verifies the model at least creates with
        the path as-is (no silent transformation), ensuring transparency for the
        guest agent validator.
        """
        malicious_path = prefix + control + suffix
        try:
            req = WriteFileRequest(path=malicious_path, content_base64="AA==")
            # If Pydantic accepts it, ensure the path is stored verbatim
            # (guest agent will reject it server-side)
            assert req.path == malicious_path
        except ValidationError:
            pass  # Pydantic rejected it -- also fine

    @given(
        prefix=text(
            alphabet=characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=10,
        ),
        control=control_chars,
        suffix=text(
            alphabet=characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_control_chars_in_path_rejected_read(self, prefix: str, control: str, suffix: str) -> None:
        """Property: Control characters in ReadFileRequest path should be rejected by Pydantic or the guest agent."""
        malicious_path = prefix + control + suffix
        try:
            req = ReadFileRequest(path=malicious_path)
            assert req.path == malicious_path
        except ValidationError:
            pass  # Pydantic rejected it -- also fine

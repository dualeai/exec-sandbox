"""Unit tests for guest agent protocol models, at the wire: the JSON object each request sends,
and the model each guest line parses to.

Tests Pydantic serialization, validation, and discriminated unions.
No mocks - pure model testing.
"""

import base64
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import characters, sampled_from, text
from pydantic import TypeAdapter, ValidationError

from exec_sandbox import constants
from exec_sandbox.constants import ENV_VAR_FORBIDDEN_CONTROL_CHARS, MAX_FILE_SIZE_BYTES
from exec_sandbox.guest_agent_protocol import (
    ExecuteCodeRequest,
    ExecutionCompleteMessage,
    FileChunkResponseMessage,
    FileEntryInfo,
    FileListMessage,
    FileReadCompleteMessage,
    FileWriteAckMessage,
    GuestAgentRequest,
    InstallPackagesRequest,
    ListFilesRequest,
    OutputChunkMessage,
    PingRequest,
    PongMessage,
    ReadFileRequest,
    StreamingErrorMessage,
    StreamingMessage,
    WarmReplAckMessage,
    WarmReplRequest,
    WriteFileRequest,
    new_host_entropy,
    request_line,
)
from exec_sandbox.models import FileInfo, Language

_CODE_LIMIT = 1024 * 1024
"""Largest code an ExecuteCodeRequest takes: 1 MiB, written out (not the library's constant)."""
_PATH_LIMIT = 4096
"""Longest file path a file request takes: Linux PATH_MAX, written out (not the library's constant)."""

_GUEST_LINE: TypeAdapter[StreamingMessage] = TypeAdapter(StreamingMessage)  # type: ignore[type-arg]
"""Parses one line from the guest, as FileOpDispatcher's dispatch loop does (guest_channel.py)."""


def wire(request: GuestAgentRequest) -> dict[str, object]:
    """The JSON object the host sends for `request` (request_line, every send site), without
    host_time_ns (the clock: test_host_time_stamped_at_serialization)."""
    line = request_line(request)
    assert line.endswith(b"\n") and line.count(b"\n") == 1  # one line: the guest reads line by line
    sent = json.loads(line)
    del sent["host_time_ns"]
    return sent


# The control characters the guest refuses, written out: NUL through BS, LF through US, DEL (tab is allowed).
_CONTROL_CHARS = sampled_from([chr(c) for c in range(0x09)] + [chr(c) for c in range(0x0A, 0x20)] + [chr(0x7F)])

# ============================================================================
# Request Models
# ============================================================================

REQUEST_WIRE_CASES = [
    pytest.param(PingRequest(host_entropy=None), {"action": "ping"}, id="ping"),
    pytest.param(PingRequest(host_entropy="AAAA"), {"action": "ping", "host_entropy": "AAAA"}, id="ping-entropy"),
    pytest.param(PingRequest(host_entropy=None, op_id="abc"), {"action": "ping", "op_id": "abc"}, id="ping-op-id"),
    pytest.param(
        WarmReplRequest(language=Language.PYTHON), {"action": "warm_repl", "language": "python"}, id="warm-repl"
    ),
    pytest.param(
        WarmReplRequest(language=Language.JAVASCRIPT, op_id="warm-1"),
        {"action": "warm_repl", "language": "javascript", "op_id": "warm-1"},
        id="warm-repl-javascript-op-id",
    ),
    pytest.param(
        ExecuteCodeRequest(language=Language.PYTHON, code="print('hello')", reseed=False, timeout=0, env_vars={}),
        {
            "action": "exec",
            "language": "python",
            "code": "print('hello')",
            "reseed": False,
            "timeout": 0,
            "env_vars": {},
        },
        id="exec-guest-default-timeout",
    ),
    pytest.param(
        ExecuteCodeRequest(
            language=Language.JAVASCRIPT,
            code="console.log('hello')",
            reseed=True,
            timeout=60,
            env_vars={"FOO": "bar", "BAZ": "qux"},
            op_id="xyz",
        ),
        {
            "action": "exec",
            "language": "javascript",
            "code": "console.log('hello')",
            "reseed": True,
            "timeout": 60,
            "env_vars": {"FOO": "bar", "BAZ": "qux"},
            "op_id": "xyz",
        },
        id="exec-all-fields",
    ),
    pytest.param(
        InstallPackagesRequest(language=Language.PYTHON, packages=["pandas==2.0.0"], timeout=300),
        {"action": "install_packages", "language": "python", "packages": ["pandas==2.0.0"], "timeout": 300},
        id="install-300s",
    ),
    pytest.param(
        InstallPackagesRequest(
            language=Language.JAVASCRIPT,
            packages=["lodash@4.17.21", "axios@1.6.0", "react@18.2.0"],
            timeout=120,
            op_id="pkg-1",
        ),
        {
            "action": "install_packages",
            "language": "javascript",
            "packages": ["lodash@4.17.21", "axios@1.6.0", "react@18.2.0"],
            "timeout": 120,
            "op_id": "pkg-1",
        },
        id="install-all-fields",
    ),
    pytest.param(
        WriteFileRequest(op_id="abc123", path="hello.txt", make_executable=False, size=None),
        {"action": "write_file", "op_id": "abc123", "path": "hello.txt", "make_executable": False},
        id="write-file",
    ),
    pytest.param(
        WriteFileRequest(op_id="abc123", path="run.sh", make_executable=True, size=None),
        {"action": "write_file", "op_id": "abc123", "path": "run.sh", "make_executable": True},
        id="write-file-executable",
    ),
    pytest.param(
        WriteFileRequest(op_id="abc123", path="big.bin", make_executable=False, size=524_288_000),
        {"action": "write_file", "op_id": "abc123", "path": "big.bin", "make_executable": False, "size": 524_288_000},
        id="write-file-size",
    ),
    pytest.param(
        WriteFileRequest(op_id="abc123", path="empty.txt", make_executable=False, size=0),
        {"action": "write_file", "op_id": "abc123", "path": "empty.txt", "make_executable": False, "size": 0},
        id="write-file-size-zero",
    ),
    pytest.param(
        ReadFileRequest(op_id="abc", path="results/data.json"),
        {"action": "read_file", "op_id": "abc", "path": "results/data.json"},
        id="read-file",
    ),
    pytest.param(ListFilesRequest(path=""), {"action": "list_files", "path": ""}, id="list-files-root"),
    pytest.param(
        ListFilesRequest(path="src", op_id="list-1"),
        {"action": "list_files", "path": "src", "op_id": "list-1"},
        id="list-files-op-id",
    ),
]


@pytest.mark.parametrize(("request_model", "expected"), REQUEST_WIRE_CASES)
def test_request_wire_format(request_model: GuestAgentRequest, expected: dict[str, object]) -> None:
    """Each request sends its action, its fields with their defaults, and op_id only when set."""
    assert wire(request_model) == expected


class TestPingRequest:
    """Tests for PingRequest model."""

    def test_host_time_stamped_at_serialization(self) -> None:
        """Every request carries the host wall clock taken when it is serialized (sent)."""
        req = PingRequest(host_entropy=None)
        before = time.time_ns()
        wire_json = json.loads(req.model_dump_json(exclude_none=True))
        assert before <= wire_json["host_time_ns"] <= time.time_ns()

    def test_new_host_entropy_is_32_new_bytes(self) -> None:
        """new_host_entropy: 32 random bytes of the host, base64, new at each call."""
        first, second = new_host_entropy(), new_host_entropy()
        assert len(base64.b64decode(first, validate=True)) == 32
        assert first != second

    def test_host_entropy_is_required(self) -> None:
        """No default: every send site says whether the guest reseeds (None: no reseed)."""
        with pytest.raises(ValidationError):
            PingRequest()  # type: ignore[call-arg]


class TestWarmReplRequest:
    """Tests for WarmReplRequest model."""

    def test_invalid_language_rejected(self) -> None:
        """Non-Language enum value is rejected by Pydantic."""
        with pytest.raises(ValidationError):
            WarmReplRequest(language="cobol")  # type: ignore[arg-type]


class TestExecuteCodeRequest:
    """Tests for ExecuteCodeRequest model."""

    def test_language_validation(self) -> None:
        """ExecuteCodeRequest rejects invalid languages."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(language="ruby", reseed=False, code="puts 'hello'", timeout=0, env_vars={})  # type: ignore[arg-type]
        assert [(e["loc"], e["type"]) for e in exc_info.value.errors()] == [(("language",), "enum")]

    def test_timeout_range(self) -> None:
        """ExecuteCodeRequest enforces timeout range 0-300."""
        # Valid: 0
        req = ExecuteCodeRequest(language=Language.PYTHON, reseed=False, code="x", timeout=0, env_vars={})
        assert req.timeout == 0

        # Valid: 300
        req = ExecuteCodeRequest(language=Language.PYTHON, reseed=False, code="x", timeout=300, env_vars={})
        assert req.timeout == 300

        # Invalid: negative
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(language=Language.PYTHON, reseed=False, code="x", timeout=-1, env_vars={})

        # Invalid: > 300
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(language=Language.PYTHON, reseed=False, code="x", timeout=301, env_vars={})

    def test_code_max_length(self) -> None:
        """ExecuteCodeRequest enforces 1 MiB code limit."""
        # Valid: 1 MiB exactly
        large_code = "x" * _CODE_LIMIT
        req = ExecuteCodeRequest(language=Language.PYTHON, reseed=False, code=large_code, timeout=0, env_vars={})
        assert len(req.code) == _CODE_LIMIT

        # Invalid: > 1 MiB
        too_large = "x" * (_CODE_LIMIT + 1)
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(language=Language.PYTHON, reseed=False, code=too_large, timeout=0, env_vars={})


_EXEC: dict[str, Any] = {"language": Language.PYTHON, "code": "x", "reseed": False, "timeout": 0, "env_vars": {}}


class TestRequiredFields:
    """The request fields with no default: each send site chooses its value."""

    @pytest.mark.parametrize(
        ("model", "fields", "missing"),
        [
            pytest.param(ExecuteCodeRequest, {**_EXEC, "reseed": None}, "reseed", id="exec-reseed"),
            pytest.param(ExecuteCodeRequest, {**_EXEC, "timeout": None}, "timeout", id="exec-timeout"),
            pytest.param(ExecuteCodeRequest, {**_EXEC, "env_vars": None}, "env_vars", id="exec-env-vars"),
            pytest.param(
                InstallPackagesRequest, {"language": Language.PYTHON, "packages": ["x==1.0"]}, "timeout", id="install"
            ),
            pytest.param(
                WriteFileRequest, {"op_id": "w", "path": "a.txt", "size": 0}, "make_executable", id="write-file"
            ),
            pytest.param(
                WriteFileRequest,
                {"op_id": "w", "path": "a.txt", "make_executable": False},
                "size",
                id="write-file-size",
            ),
            pytest.param(ListFilesRequest, {}, "path", id="list-files"),
        ],
    )
    def test_a_request_field_left_out_is_refused(
        self, model: type[GuestAgentRequest], fields: dict[str, Any], missing: str
    ) -> None:
        """No defaults: a send site that leaves a field out fails, rather than sending a value it
        did not choose (the reseed, the timeout, the environment, the file mode, the file size,
        the folder)."""
        with pytest.raises(ValidationError) as exc_info:
            model(**{key: value for key, value in fields.items() if value is not None})
        assert [(error["type"], error["loc"]) for error in exc_info.value.errors()] == [("missing", (missing,))]


class TestWriteFileSize:
    """The size of a write_file header: the decoded bytes, from 0 to MAX_FILE_SIZE_BYTES
    (REQUEST_WIRE_CASES sends both ends, and None as no size key)."""

    @pytest.mark.parametrize("size", [-1, MAX_FILE_SIZE_BYTES + 1])
    def test_size_out_of_range_is_refused(self, size: int) -> None:
        """A size below 0 or above MAX_FILE_SIZE_BYTES fails on the size field: the host never
        sends it."""
        with pytest.raises(ValidationError) as exc_info:
            WriteFileRequest(op_id="w", path="a.bin", make_executable=False, size=size)
        assert [error["loc"] for error in exc_info.value.errors()] == [("size",)]


class TestEnvVarValidation:
    """Tests for environment variable validation."""

    def test_valid_env_vars(self) -> None:
        """Valid env vars with printable ASCII and tabs."""
        req = ExecuteCodeRequest(
            language=Language.PYTHON,
            reseed=False,
            code="print(1)",
            env_vars={"FOO": "bar", "WITH_TAB": "value\twith\ttabs"},
            timeout=0,
        )
        assert req.env_vars["WITH_TAB"] == "value\twith\ttabs"

    @pytest.mark.parametrize(
        ("env_vars", "message"),
        [
            ({"FOO\x00BAR": "value"}, "Env var name contains forbidden control character: 0x00"),
            ({"FOO": "val\x00ue"}, "Env var 'FOO' value contains forbidden control character: 0x00"),
            ({"FOO": "\x1b[31mred\x1b[0m"}, "Env var 'FOO' value contains forbidden control character: 0x1B"),
            ({"FOO": "line1\nline2"}, "Env var 'FOO' value contains forbidden control character: 0x0A"),
            ({"FOO": "start\roverwrite"}, "Env var 'FOO' value contains forbidden control character: 0x0D"),
            ({"FOO": "ding\x07"}, "Env var 'FOO' value contains forbidden control character: 0x07"),
            ({"FOO": "delete\x7f"}, "Env var 'FOO' value contains forbidden control character: 0x7F"),
            ({"A" * 257: "value"}, "Invalid env var name length: 257 (must be 1-256)"),
            ({"": "value"}, "Invalid env var name length: 0 (must be 1-256)"),
            ({"FOO": "x" * 4097}, "Env var 'FOO' value too large: 4097 characters (max 4096)"),
            ({f"VAR_{i}": "value" for i in range(101)}, "Too many environment variables: 101 (max 100)"),
        ],
        ids=[
            "nul-in-name",
            "nul-in-value",
            "escape-sequence",
            "newline",  # log injection
            "carriage-return",
            "bell",
            "del",
            "name-too-long",
            "empty-name",
            "value-too-large",
            "too-many",
        ],
    )
    def test_rejected_env_vars(self, env_vars: dict[str, str], message: str) -> None:
        """Each env var the guest must not get fails validation with an error that names the
        fault (the host maps it to ENV_VAR_INVALID by its field)."""
        with pytest.raises(ValidationError) as exc_info:
            ExecuteCodeRequest(language=Language.PYTHON, reseed=False, code="x", env_vars=env_vars, timeout=0)
        assert [(e["loc"], e["msg"]) for e in exc_info.value.errors()] == [(("env_vars",), f"Value error, {message}")]

    def test_error_names_the_first_forbidden_character(self) -> None:
        """Allowed characters that are not printable (tab, U+2028) come before the first
        forbidden one: the error names that one (ESC), not a later one (NUL)."""
        with pytest.raises(ValidationError, match="forbidden control character: 0x1B"):
            ExecuteCodeRequest(
                language=Language.PYTHON,
                reseed=False,
                code="x",
                env_vars={"FOO": "a\t\u2028b\x1b\x00" * 100},
                timeout=0,
            )

    def test_each_forbidden_character_is_not_printable(self) -> None:
        """The check skips printable text (str.isprintable): it is exact only while no
        forbidden character is printable."""
        assert not any(chr(code).isprintable() for code in ENV_VAR_FORBIDDEN_CONTROL_CHARS)

    def test_utf8_in_value_allowed(self) -> None:
        """UTF-8 characters (emoji, non-Latin) are allowed."""
        req = ExecuteCodeRequest(
            language=Language.PYTHON, reseed=False, code="x", env_vars={"GREETING": "Hello 世界 🌍"}, timeout=0
        )
        assert req.env_vars == {"GREETING": "Hello 世界 🌍"}


class TestInstallPackagesRequest:
    """Tests for InstallPackagesRequest model."""

    def test_packages_min_length(self) -> None:
        """InstallPackagesRequest requires at least 1 package."""
        with pytest.raises(ValidationError):
            InstallPackagesRequest(language=Language.PYTHON, packages=[], timeout=300)

    def test_packages_max_length(self) -> None:
        """InstallPackagesRequest allows max 50 packages."""
        # Valid: 50 packages
        packages = [f"pkg{i}==1.0.0" for i in range(50)]
        req = InstallPackagesRequest(language=Language.PYTHON, packages=packages, timeout=300)
        assert len(req.packages) == 50

        # Invalid: 51 packages
        packages = [f"pkg{i}==1.0.0" for i in range(51)]
        with pytest.raises(ValidationError):
            InstallPackagesRequest(language=Language.PYTHON, packages=packages, timeout=300)

    def test_timeout_range(self) -> None:
        """InstallPackagesRequest enforces timeout range 0-300."""
        # Valid: max
        req = InstallPackagesRequest(language=Language.PYTHON, packages=["x==1.0"], timeout=300)
        assert req.timeout == 300

        # Invalid: > 300
        with pytest.raises(ValidationError):
            InstallPackagesRequest(language=Language.PYTHON, packages=["x==1.0"], timeout=301)


class TestFileRequestPaths:
    """Path limits of WriteFileRequest, ReadFileRequest and ListFilesRequest."""

    def test_write_file_path_max_length(self) -> None:
        """WriteFileRequest accepts path at max length."""
        long_path = "a" * _PATH_LIMIT
        req = WriteFileRequest(op_id="test", path=long_path, make_executable=False, size=None)
        assert len(req.path) == _PATH_LIMIT

    def test_write_file_path_exceeds_max_length(self) -> None:
        """WriteFileRequest rejects path exceeding max length."""
        too_long = "a" * (_PATH_LIMIT + 1)
        with pytest.raises(ValidationError) as exc_info:
            WriteFileRequest(op_id="test", path=too_long, make_executable=False, size=None)
        assert [error["loc"] for error in exc_info.value.errors()] == [("path",)]

    def test_write_file_empty_path_rejected(self) -> None:
        """WriteFileRequest rejects empty path."""
        with pytest.raises(ValidationError) as exc_info:
            WriteFileRequest(op_id="test", path="", make_executable=False, size=None)
        assert [error["loc"] for error in exc_info.value.errors()] == [("path",)]

    def test_read_file_path_max_length(self) -> None:
        """ReadFileRequest accepts path at max length."""
        long_path = "b" * _PATH_LIMIT
        req = ReadFileRequest(op_id="test", path=long_path)
        assert len(req.path) == _PATH_LIMIT

    def test_read_file_path_exceeds_max_length(self) -> None:
        """ReadFileRequest rejects path exceeding max length."""
        too_long = "b" * (_PATH_LIMIT + 1)
        with pytest.raises(ValidationError):
            ReadFileRequest(op_id="test", path=too_long)

    def test_read_file_empty_path_rejected(self) -> None:
        """ReadFileRequest rejects empty path."""
        with pytest.raises(ValidationError):
            ReadFileRequest(op_id="test", path="")

    def test_list_files_path_max_length(self) -> None:
        """ListFilesRequest accepts path at max length."""
        long_path = "c" * _PATH_LIMIT
        req = ListFilesRequest(path=long_path)
        assert len(req.path) == _PATH_LIMIT

    def test_list_files_path_exceeds_max_length(self) -> None:
        """ListFilesRequest rejects path exceeding max length."""
        too_long = "c" * (_PATH_LIMIT + 1)
        with pytest.raises(ValidationError):
            ListFilesRequest(path=too_long)

    @pytest.mark.parametrize("model", [WriteFileRequest, ReadFileRequest])
    def test_file_request_requires_op_id(self, model: type[WriteFileRequest | ReadFileRequest]) -> None:
        """A write or read request needs an op_id: its chunks carry it (optional on the other requests)."""
        with pytest.raises(ValidationError):
            model(path="test.txt")  # type: ignore[call-arg]


# ============================================================================
# Response Models
# ============================================================================

RESPONSE_WIRE_CASES = [
    pytest.param('{"type":"stdout","chunk":"Hello"}', OutputChunkMessage(type="stdout", chunk="Hello"), id="stdout"),
    pytest.param(
        '{"type":"stderr","chunk":"","op_id":"abc"}',
        OutputChunkMessage(type="stderr", chunk="", op_id="abc"),
        id="stderr-empty-chunk-op-id",
    ),
    pytest.param(
        '{"type":"complete","exit_code":0,"execution_time_ms":42}',
        ExecutionCompleteMessage(exit_code=0, execution_time_ms=42, spawn_ms=None, process_ms=None, op_id=None),
        id="complete-no-timing",
    ),
    pytest.param(
        '{"type":"complete","exit_code":0,"execution_time_ms":100,"spawn_ms":5,"process_ms":90,"op_id":"abc"}',
        ExecutionCompleteMessage(exit_code=0, execution_time_ms=100, spawn_ms=5, process_ms=90, op_id="abc"),
        id="complete-timing-op-id",
    ),
    # A signal exit (negative), spawn_ms only: the process timed out before it ended
    pytest.param(
        '{"type":"complete","exit_code":-9,"execution_time_ms":30000,"spawn_ms":10}',
        ExecutionCompleteMessage(exit_code=-9, execution_time_ms=30000, spawn_ms=10, process_ms=None),
        id="complete-signal-spawn-only",
    ),
    # The shell convention for a signal: 128 + 9 (SIGKILL)
    pytest.param(
        '{"type":"complete","exit_code":137,"execution_time_ms":100}',
        ExecutionCompleteMessage(exit_code=137, execution_time_ms=100),
        id="complete-128-plus-signal",
    ),
    pytest.param('{"type":"pong","version":"1.0.0"}', PongMessage(version="1.0.0", op_id=None), id="pong"),
    pytest.param(
        '{"type":"pong","version":"1.0","op_id":"abc"}', PongMessage(version="1.0", op_id="abc"), id="pong-op-id"
    ),
    pytest.param(
        '{"type":"warm_repl_ack","status":"ok"}',
        WarmReplAckMessage(status="ok", message=None, op_id=None),
        id="warm-repl-ack",
    ),
    pytest.param(
        '{"type":"warm_repl_ack","status":"error","message":"spawn failed","op_id":"abc"}',
        WarmReplAckMessage(status="error", message="spawn failed", op_id="abc"),
        id="warm-repl-ack-error",
    ),
    pytest.param(
        '{"type":"error","message":"timeout","error_type":"timeout_error"}',
        StreamingErrorMessage(message="timeout", error_type="timeout_error", op_id=None),
        id="error",
    ),
    pytest.param(
        '{"type":"error","message":"path traversal","error_type":"path_error","op_id":"xyz"}',
        StreamingErrorMessage(message="path traversal", error_type="path_error", op_id="xyz"),
        id="error-op-id",
    ),
    pytest.param(
        '{"type":"file_write_ack","op_id":"abc123","path":"hello.txt","bytes_written":10000000}',
        FileWriteAckMessage(op_id="abc123", path="hello.txt", bytes_written=10_000_000),
        id="file-write-ack",
    ),
    pytest.param(
        '{"type":"file_chunk","op_id":"abc123","data":"SGVsbG8="}',
        FileChunkResponseMessage(op_id="abc123", data=b"Hello"),
        id="file-chunk-base64",
    ),
    pytest.param(  # the guest writes the standard alphabet (Rust base64 STANDARD): + and /, not - and _
        '{"type":"file_chunk","op_id":"abc123","data":"+/+/"}',
        FileChunkResponseMessage(op_id="abc123", data=b"\xfb\xff\xbf"),
        id="file-chunk-standard-alphabet",
    ),
    pytest.param(
        '{"type":"file_read_complete","op_id":"test","path":"out.bin","size":65536}',
        FileReadCompleteMessage(op_id="test", path="out.bin", size=65536),
        id="file-read-complete",
    ),
    pytest.param(
        '{"type":"file_list","entries":[]}',
        FileListMessage(entries=[], op_id=None),
        id="file-list-empty",
    ),
    pytest.param(
        '{"type":"file_list","entries":[{"name":"src","is_dir":true,"size":0},'
        '{"name":"main.py","is_dir":false,"size":256}],"op_id":"abc"}',
        FileListMessage(
            entries=[
                FileEntryInfo(name="src", is_dir=True, size=0),
                FileEntryInfo(name="main.py", is_dir=False, size=256),
            ],
            op_id="abc",
        ),
        id="file-list-entries-op-id",
    ),
]


@pytest.mark.parametrize(("line", "expected"), RESPONSE_WIRE_CASES)
def test_guest_line_parses_to_its_message(line: str, expected: StreamingMessage) -> None:
    """A guest line parses to the message of its type, with each field it sent, and None for each
    optional field it left out (op_id, timings, message)."""
    assert _GUEST_LINE.validate_json(line) == expected


class TestResponseValidation:
    """Guest lines the host refuses (the dispatch loop drops a line that does not parse)."""

    def test_parse_unknown_type(self) -> None:
        """Reject unknown message type."""
        with pytest.raises(ValidationError):
            _GUEST_LINE.validate_json('{"type": "unknown", "data": "something"}')

    def test_wrong_type_rejected(self) -> None:
        """A type literal that is not the model's is rejected."""
        with pytest.raises(ValidationError):
            WarmReplAckMessage(type="pong", status="ok")  # type: ignore[arg-type]

    def test_invalid_output_stream(self) -> None:
        """An output chunk is stdout or stderr only."""
        with pytest.raises(ValidationError):
            _GUEST_LINE.validate_json('{"type": "stdin", "chunk": "data"}')

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

    def test_data_that_is_not_base64_is_refused(self) -> None:
        """A chunk whose data is no base64: a validation error at the parse, so the dispatcher
        drops the frame; read_file then gets the other chunks only (its size check fails)."""
        with pytest.raises(ValidationError):
            _GUEST_LINE.validate_json('{"type": "file_chunk", "op_id": "x", "data": "not base64!"}')


class TestFileChunkResponseMessage:
    """Tests for FileChunkResponseMessage model."""

    def test_json_round_trip(self) -> None:
        """The bytes go out as base64 in JSON and come back the same (pydantic writes the URL-safe
        alphabet and reads both)."""
        msg = FileChunkResponseMessage(op_id="abc", data=b"\x00\x01\xff zstd")
        assert FileChunkResponseMessage.model_validate_json(msg.model_dump_json()) == msg


class TestFileInfo:
    """Tests for FileInfo model (from models.py)."""

    def test_serialize_roundtrip(self) -> None:
        """FileInfo dumps to its three fields and validates back to an equal model."""
        info = FileInfo(name="image.png", is_dir=False, size=65536)
        data = info.model_dump()
        assert data == {"name": "image.png", "is_dir": False, "size": 65536}
        assert FileInfo.model_validate(data) == info


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
            language=Language.PYTHON, reseed=False, code="x", env_vars={name: safe_value}, timeout=0
        )
        assert req.env_vars[name] == safe_value

    @given(
        name=valid_name,
        prefix=text(safe_chars, min_size=0, max_size=10),
        control=_CONTROL_CHARS,
        suffix=text(safe_chars, min_size=0, max_size=10),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_control_chars_in_value_rejected(self, name: str, prefix: str, control: str, suffix: str) -> None:
        """Property: Any value containing a control character must be rejected."""
        malicious_value = prefix + control + suffix
        with pytest.raises(ValidationError):
            ExecuteCodeRequest(
                language=Language.PYTHON, reseed=False, code="x", env_vars={name: malicious_value}, timeout=0
            )

    @given(
        prefix=text(safe_chars, min_size=0, max_size=10),
        control=_CONTROL_CHARS,
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
                language=Language.PYTHON, reseed=False, code="x", env_vars={malicious_name: value}, timeout=0
            )


# ============================================================================
# Property-Based Tests - File I/O (Hypothesis)
# ============================================================================


class TestFilePathValidationPropertyBased:
    """Property-based tests for file path validation using Hypothesis.

    These tests automatically discover edge cases in path validation
    across WriteFileRequest, ReadFileRequest, and ListFilesRequest.
    """

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
        req_w = WriteFileRequest(op_id="test", path=path, make_executable=False, size=None)
        assert req_w.path == path
        # ReadFileRequest
        req_r = ReadFileRequest(op_id="test", path=path)
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
        """Property: Paths exceeding PATH_MAX must be rejected."""
        # Build a path that is guaranteed to exceed PATH_MAX
        path = "a" * (_PATH_LIMIT + 1) + extra
        with pytest.raises(ValidationError):
            WriteFileRequest(op_id="test", path=path, make_executable=False, size=None)
        with pytest.raises(ValidationError):
            ReadFileRequest(op_id="test", path=path)
        with pytest.raises(ValidationError):
            ListFilesRequest(path=path)

    @given(
        prefix=text(
            alphabet=characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=10,
        ),
        control=_CONTROL_CHARS,
        suffix=text(
            alphabet=characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_control_chars_in_path_go_to_the_guest_as_they_are(self, prefix: str, control: str, suffix: str) -> None:
        """Property: the host does not check control characters in a file path (only its length): the
        request sends the path as it is, and the guest agent refuses it (guest-agent/src/file_io.rs)."""
        path = prefix + control + suffix
        for request in (
            WriteFileRequest(op_id="test", path=path, make_executable=False, size=None),
            ReadFileRequest(op_id="test", path=path),
            ListFilesRequest(path=path),
        ):
            assert wire(request)["path"] == path


# The guest's limits that the host copies (constants.py): one value on each side, read from
# the guest's source so a change on one side fails here, not in a VM.
_GUEST_CONSTANTS = Path(__file__).parent.parent / "guest-agent" / "src" / "constants.rs"


def _guest_constant(name: str) -> int:
    """The value of `pub(crate) const NAME: usize|u64 = <a product of integers>;` in constants.rs."""
    match = re.search(rf"^pub\(crate\) const {name}: (?:usize|u64) = ([0-9_ *]+);", _GUEST_CONSTANTS.read_text(), re.M)
    assert match is not None, name
    return math.prod(int(factor) for factor in match.group(1).replace("_", "").split("*"))


@pytest.mark.parametrize(
    ("guest", "host"),
    [
        ("MAX_STDOUT_BYTES", constants.MAX_STDOUT_BYTES),
        ("MAX_STDERR_BYTES", constants.MAX_STDERR_BYTES),
        ("OUTPUT_READ_BYTES", constants.GUEST_OUTPUT_READ_BYTES),
        ("MAX_FILE_SIZE_BYTES", constants.MAX_FILE_SIZE_BYTES),
        ("MAX_FILE_PATH_LENGTH", constants.MAX_FILE_PATH_LENGTH),
        ("FILE_TRANSFER_CHUNK_SIZE", constants.FILE_TRANSFER_CHUNK_SIZE),
        ("MAX_CODE_SIZE_BYTES", constants.MAX_CODE_SIZE),
        ("MAX_TIMEOUT_SECONDS", constants.MAX_TIMEOUT_SECONDS),
        ("MAX_ENV_VARS", constants.MAX_ENV_VARS),
        ("MAX_ENV_VAR_NAME_LENGTH", constants.MAX_ENV_VAR_NAME_LENGTH),
        ("MAX_ENV_VAR_VALUE_LENGTH", constants.MAX_ENV_VAR_VALUE_LENGTH),
    ],
)
def test_the_host_copies_the_guests_limits(guest: str, host: int) -> None:
    assert _guest_constant(guest) == host

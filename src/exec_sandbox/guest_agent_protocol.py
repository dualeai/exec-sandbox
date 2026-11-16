"""Guest agent communication protocol models.

Defines request/response types for host-guest communication via virtio-serial.
Protocol: JSON newline-delimited over dual-port channel (command + event).

Security: the guest agent validates every field again (guest-agent/src/validation.rs,
file_io.rs, packages.rs).

Releases: a saved VM keeps the guest agent of its release and can load on an older or a newer
host release, with no version check. So each change is additive in both directions:
- a new request field is optional for the guest, and an older guest ignores it: the call must
  stay correct without its effect (request_line drops a None field);
- the reply models ignore unknown fields (a new reply field is optional here); no new reply
  type and no new error_type (an older host fails them);
- before file_end, a write gets no reply but a terminal error.
"""

from __future__ import annotations

import base64
import os
import re
import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from exec_sandbox.constants import (
    ENV_VAR_FORBIDDEN_CONTROL_CHARS,
    MAX_CODE_SIZE,
    MAX_ENV_VAR_NAME_LENGTH,
    MAX_ENV_VAR_VALUE_LENGTH,
    MAX_ENV_VARS,
    MAX_FILE_PATH_LENGTH,
    MAX_FILE_SIZE_BYTES,
)
from exec_sandbox.models import Language  # noqa: TC001 - Required at runtime for Pydantic

_FORBIDDEN_CHAR = re.compile(
    "[" + "".join(re.escape(chr(code)) for code in sorted(ENV_VAR_FORBIDDEN_CONTROL_CHARS)) + "]"
)
"""One of ENV_VAR_FORBIDDEN_CONTROL_CHARS."""


def _forbidden_char(text: str) -> int | None:
    """The code of the first ENV_VAR_FORBIDDEN_CONTROL_CHARS character in `text`, or None.
    Each of them is a control character, so printable text has none: str.isprintable() checks
    that in C first (a loop over each character in Python took 5.5 ms of event loop time for
    100 values of 4096 characters)."""
    if text.isprintable():
        return None
    found = _FORBIDDEN_CHAR.search(text)
    return None if found is None else ord(found.group())


# ============================================================================
# Request Models
# ============================================================================


class GuestAgentRequest(BaseModel):
    """Base class for all guest agent requests."""

    action: str = Field(description="Action to perform")
    op_id: str | None = Field(default=None, description="Operation ID for request-response correlation")

    @computed_field
    @property
    def host_time_ns(self) -> int:
        """Host wall clock when the request is serialized (sent); the guest steps its clock
        forward to it when it lags by more than 100 ms (never back)."""
        return time.time_ns()


def new_host_entropy() -> str:
    """32 random bytes of the host, base64: a PingRequest's host_entropy."""
    return base64.b64encode(os.urandom(32)).decode()


def request_line(request: GuestAgentRequest) -> bytes:
    """`request` as the guest agent reads it: its JSON with no null field, then a newline (every
    host send site: DualPortChannel, QemuVM's file transfers)."""
    return request.model_dump_json(by_alias=False, exclude_none=True).encode() + b"\n"


class PingRequest(GuestAgentRequest):
    """Health check request.

    Response: PongMessage with version field; with host_entropy, a StreamingErrorMessage
    (io_error) instead when the guest cannot reseed its kernel random source.
    """

    action: Literal["ping"] = Field(default="ping")  # type: ignore[assignment]
    host_entropy: str | None = Field(
        description=(
            "32 random bytes of the host, base64 (new_host_entropy): the guest mixes them into its "
            "kernel random source and reseeds it, and answers an error, not a pong, when it cannot. "
            "None: no host bytes (the guest reseeds from its own input pool, as before each command)"
        )
    )


class WarmReplRequest(GuestAgentRequest):
    """Warm the REPL of a language: the guest takes its live REPL, or starts one and waits
    for its ready line, then runs a no-op on it with the reseed flag (ExecuteCodeRequest.reseed),
    so the first exec of a user is not the REPL's first, and a copy of a saved state runs its
    reseed warm.

    Response: WarmReplAckMessage (status "ok" or "error").
    """

    action: Literal["warm_repl"] = Field(default="warm_repl")  # type: ignore[assignment]
    language: Language = Field(description="Language REPL to pre-warm")


class ExecuteCodeRequest(GuestAgentRequest):
    """Execute code in guest VM.

    Guest agent enforces:
    - Code size limit: 1 MiB
    - Timeout limit: 300s
    - Env var limits: 100 vars, names of 1-256 bytes, values of at most 4096 bytes (UTF-8;
      the host check counts characters)
    - Blocked env vars: LD_PRELOAD, NODE_OPTIONS, PATH, etc. (security)

    Response: OutputChunkMessage messages, then ExecutionCompleteMessage or StreamingErrorMessage.
    """

    action: Literal["exec"] = Field(default="exec")  # type: ignore[assignment]
    language: Language = Field(description="Programming language for execution")
    code: str = Field(max_length=MAX_CODE_SIZE, description="Code to execute (max 1 MiB)")

    # Note: empty-code and null-byte checks live in guest-agent validation.rs.
    # The host translates the guest's code_error response to InvalidInputError (CODE_INVALID)
    # via guest_error_to_exception. No Pydantic validator here.

    reseed: bool = Field(
        description="True: the REPL reseeds its random generators from the kernel before it runs the "
        "code (Python: ssl and the module functions of random; JavaScript: BoringSSL's generator of the "
        "REPL thread; bash: $RANDOM)"
    )

    timeout: int = Field(ge=0, le=300, description="Execution timeout in seconds (0 = the default, 300 s; max 300 s)")
    env_vars: dict[str, str] = Field(
        description="Environment variables (max 100, blocked vars enforced by guest agent)",
    )

    @field_validator("env_vars")
    @classmethod
    def validate_env_vars(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate environment variable names and values.

        Security: Rejects control characters to prevent terminal escape injection,
        log injection, and protocol manipulation attacks.
        """
        if len(v) > MAX_ENV_VARS:
            raise ValueError(f"Too many environment variables: {len(v)} (max {MAX_ENV_VARS})")

        for key, value in v.items():
            # Validate name length
            if not key or len(key) > MAX_ENV_VAR_NAME_LENGTH:
                raise ValueError(f"Invalid env var name length: {len(key)} (must be 1-{MAX_ENV_VAR_NAME_LENGTH})")

            # Validate value length
            if len(value) > MAX_ENV_VAR_VALUE_LENGTH:
                raise ValueError(
                    f"Env var '{key}' value too large: {len(value)} characters (max {MAX_ENV_VAR_VALUE_LENGTH})"
                )

            # Check for control characters in name, then in value (the first one found)
            if (code := _forbidden_char(key)) is not None:
                raise ValueError(f"Env var name contains forbidden control character: 0x{code:02X}")
            if (code := _forbidden_char(value)) is not None:
                raise ValueError(f"Env var '{key}' value contains forbidden control character: 0x{code:02X}")

        return v


class InstallPackagesRequest(GuestAgentRequest):
    """Install packages with `uv pip install` (Python) or `bun add` (JavaScript); the
    guest refuses raw.

    Guest agent enforces:
    - Package count limit: 50
    - Package spec length: 214 bytes, name and version (npm's name limit)
    - Version specifier required: pandas==2.0.0, lodash@4.17.21
    - Path traversal protection: no /, .., \\
    - Timeout: 0 = the default, 300 s (the host model caps it at 300 s; the guest checks no
      maximum)

    Response: OutputChunkMessage messages, then ExecutionCompleteMessage or StreamingErrorMessage.
    """

    action: Literal["install_packages"] = Field(default="install_packages")  # type: ignore[assignment]
    language: Language = Field(
        description="Programming language (python: uv pip, javascript: bun add; raw has no package manager: refused)"
    )
    packages: list[str] = Field(
        min_length=1,
        max_length=50,
        description="Package list with version specifiers (e.g., ['pandas==2.0.0', 'lodash@4.17.21'])",
    )
    timeout: int = Field(
        ge=0, le=300, description="Installation timeout in seconds (0 = the default, 300 s; max 300 s)"
    )


# ============================================================================
# File I/O Request Models (Streaming Chunked Protocol)
# ============================================================================


class WriteFileRequest(GuestAgentRequest):
    """Header for streaming file write to the sandbox.

    Initiates a multi-message write sequence:
    1. Host sends WriteFileRequest (this header with op_id, and size when the host knows it)
    2. Host sends file_chunk frames (op_id, data: base64 of zstd-compressed chunks)
    3. Host sends a file_end frame (op_id; signals end of stream)
    4. Guest responds with FileWriteAckMessage

    Before file_end, the guest sends nothing for this op_id but a terminal error
    (StreamingErrorMessage: a path, I/O, space or protocol refusal, or a stall); QemuVM.write_file
    stops at it. Every guest release keeps to this: QemuVM.write_file raises GUEST_PROTOCOL for any
    other reply there.

    QemuVM writes the file_chunk and file_end frames itself (no model): at most one
    file_chunk per 128 KB read of the source (none while the compressor buffers), one more
    for the compressor's flush, then file_end; guest-agent/src/types.rs
    GuestCommand::FileChunk/FileEnd parse them.

    Guest agent enforces:
    - Path length: PATH_MAX 4096 bytes, NAME_MAX 255 bytes per component
    - Path traversal: no escape from /home/user
    - File size: max 500 MiB (streamed in zstd chunks: memory does not grow with the file)
    - Free space: with a size above 0, the home disk has room for it, else an io_error: at the
      header when its free-space count is too small, else when its reservation of that room
      fails (at the next file_chunk or at file_end). The reservation holds until the commit
      (which frees the part the file does not use) or until the write fails
    - Permissions: 0o644, or 0o755 with make_executable=True

    Response: FileWriteAckMessage (after file_end).
    """

    action: Literal["write_file"] = Field(default="write_file")  # type: ignore[assignment]
    op_id: str = Field(description="Operation ID (UUID4 hex) for correlating chunks")  # type: ignore[assignment]
    path: str = Field(
        min_length=1,
        max_length=MAX_FILE_PATH_LENGTH,
        description="Relative path in sandbox (NAME_MAX 255 bytes per component)",
    )
    make_executable: bool = Field(description="Set executable permission (0o755 vs 0o644)")
    size: int | None = Field(
        ge=0,
        le=MAX_FILE_SIZE_BYTES,
        description=(
            "Bytes of the file after decompression, when the host knows them before the first "
            "file_chunk: what is left in a seekable source. None: not known (a non-seekable "
            "source, or one with no end to seek to, as a procfs file). The guest checks and "
            "reserves this room; the file still gets the bytes that come before file_end"
        ),
    )


class ReadFileRequest(GuestAgentRequest):
    """Read a file from the sandbox (streaming response).

    Guest responds with:
    1. FileChunkResponseMessage messages (zstd-compressed chunks)
    2. FileReadCompleteMessage (signals end of stream with metadata)

    Response: Streaming FileChunkResponseMessage + FileReadCompleteMessage.
    """

    action: Literal["read_file"] = Field(default="read_file")  # type: ignore[assignment]
    op_id: str = Field(description="Operation ID (UUID4 hex) to match response chunks")  # type: ignore[assignment]
    path: str = Field(
        min_length=1,
        max_length=MAX_FILE_PATH_LENGTH,
        description="Relative path in sandbox (NAME_MAX 255 bytes per component)",
    )


class ListFilesRequest(GuestAgentRequest):
    """List files in a sandbox directory.

    Response: FileListMessage.
    """

    action: Literal["list_files"] = Field(default="list_files")  # type: ignore[assignment]
    path: str = Field(max_length=MAX_FILE_PATH_LENGTH, description="Relative path (empty for sandbox root)")


# ============================================================================
# Streaming Response Models
# ============================================================================


class GuestAgentResponse(BaseModel):
    """Base class for all guest agent responses.

    Every response carries an optional op_id for request-response correlation.
    The guest agent echoes back the op_id from the request.
    """

    op_id: str | None = Field(default=None, description="Operation ID echoed from the request")


class OutputChunkMessage(GuestAgentResponse):
    """Streaming output chunk from code execution.

    The guest batches output and flushes it:
    - every 50ms (output shows almost at once), and at once for the first output
      after 50ms with no output,
    - or once 64KB has accumulated (memory stays bounded),
    - or when the process ends (final flush).

    A full write queue (a bounded channel) makes the guest wait: backpressure.
    """

    type: Literal["stdout", "stderr"] = Field(description="Output stream type")
    chunk: str = Field(
        max_length=10_000_000,
        description="Output chunk (batched over 50ms window, max 64KB per flush)",
    )


class ExecutionCompleteMessage(GuestAgentResponse):
    """Last message of an exec or an install_packages that ran to its end.

    Sent after the command's output (OutputChunkMessage): an exec streams its
    output while it runs; install_packages sends its output once the install
    ends. A refused or failed command (validation, output limit, timeout) ends with
    StreamingErrorMessage instead. Other commands end with their own reply
    (PongMessage, WarmReplAckMessage, the file messages).
    """

    type: Literal["complete"] = "complete"
    exit_code: int = Field(description="Process exit code (0=success)")
    execution_time_ms: int = Field(description="Total execution time in milliseconds")
    spawn_ms: int | None = Field(
        default=None,
        description=(
            "Time to get the REPL process in milliseconds (guest-reported): a new REPL's start up "
            "to its ready line, ~0 ms when one runs."
        ),
    )
    process_ms: int | None = Field(
        default=None,
        description=(
            "Time from the code's write to the REPL to the end of its output, in milliseconds "
            "(guest-reported); None for install_packages"
        ),
    )


class StreamingErrorMessage(GuestAgentResponse):
    """Error message during streaming execution."""

    type: Literal["error"] = "error"
    message: str = Field(description="Error message")
    error_type: str = Field(description="Error classification")


class PongMessage(GuestAgentResponse):
    """Response to ping request."""

    type: Literal["pong"] = "pong"
    version: str = Field(description="Guest agent version")


class WarmReplAckMessage(GuestAgentResponse):
    """Acknowledgement for REPL pre-warming."""

    type: Literal["warm_repl_ack"] = "warm_repl_ack"
    status: str  # "ok" or "error"
    message: str | None = None


# ============================================================================
# File I/O Response Models (Streaming Chunked Protocol)
# ============================================================================


class FileWriteAckMessage(GuestAgentResponse):
    """Acknowledgment after completing a streaming file write."""

    type: Literal["file_write_ack"] = "file_write_ack"
    op_id: str = Field(description="Operation ID echoed from WriteFileRequest")  # type: ignore[assignment]
    path: str = Field(description="Path of written file (relative to sandbox root)")
    bytes_written: int = Field(description="Number of bytes written (decompressed)")


class FileChunkResponseMessage(GuestAgentResponse):
    """A chunk of zstd-compressed file data (guest → host).

    Part of a streaming file read response. Each chunk contains zstd-compressed
    data belonging to a single zstd frame, base64 in JSON: pydantic decodes it while it
    parses the message (115 -> 40 us per 128 KB chunk on the event loop, Apple M).
    """

    model_config = ConfigDict(val_json_bytes="base64", ser_json_bytes="base64")

    type: Literal["file_chunk"] = "file_chunk"
    op_id: str = Field(description="Operation ID from ReadFileRequest")  # type: ignore[assignment]
    data: bytes = Field(description="zstd-compressed chunk (base64 in JSON)")


class FileReadCompleteMessage(GuestAgentResponse):
    """Signals end of a streaming file read (guest → host).

    Sent after all FileChunkResponseMessage messages for a read operation.
    """

    type: Literal["file_read_complete"] = "file_read_complete"
    op_id: str = Field(description="Operation ID from ReadFileRequest")  # type: ignore[assignment]
    path: str = Field(description="Path of read file (relative to sandbox root)")
    size: int = Field(description="Original file size in bytes (before compression)")


class FileEntryInfo(BaseModel):
    """File entry in a directory listing (wire format)."""

    name: str = Field(description="File or directory name")
    is_dir: bool = Field(description="True if entry is a directory")
    size: int = Field(
        description="Bytes the file takes on disk, at most its length: min(length, allocated blocks x 512); "
        "0 for directories"
    )


class FileListMessage(GuestAgentResponse):
    """Directory listing response from the sandbox."""

    type: Literal["file_list"] = "file_list"
    entries: list[FileEntryInfo] = Field(description="Directory entries")


# Discriminated union for streaming messages
StreamingMessage = Annotated[
    OutputChunkMessage
    | ExecutionCompleteMessage
    | PongMessage
    | WarmReplAckMessage
    | StreamingErrorMessage
    | FileWriteAckMessage
    | FileChunkResponseMessage
    | FileReadCompleteMessage
    | FileListMessage,
    Field(discriminator="type"),
]

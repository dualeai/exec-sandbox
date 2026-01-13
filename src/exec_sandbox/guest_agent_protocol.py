"""Guest agent communication protocol models.

Defines request/response types for host-guest communication via TCP.
Protocol: JSON newline-delimited, synchronous request-response.

Security: All fields validated by guest agent (see guest-agent/src/main.rs).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from exec_sandbox.models import Language  # noqa: TC001 - Required at runtime for Pydantic

# ============================================================================
# Request Models
# ============================================================================


class GuestAgentRequest(BaseModel):
    """Base class for all guest agent requests."""

    action: str = Field(description="Action to perform")


class PingRequest(GuestAgentRequest):
    """Health check request.

    Response: PongMessage with version field.
    """

    action: Literal["ping"] = Field(default="ping")  # type: ignore[assignment]


class ExecuteCodeRequest(GuestAgentRequest):
    """Execute code in guest VM.

    Guest agent enforces:
    - Code size limit: 1MB
    - Timeout limit: 300s
    - Env var limits: 100 vars, 256 char names, 4096 char values
    - Blocked env vars: LD_PRELOAD, NODE_OPTIONS, PATH, etc. (security)

    Response: Streaming messages (OutputChunkMessage, ExecutionCompleteMessage).
    """

    action: Literal["exec"] = Field(default="exec")  # type: ignore[assignment]
    language: Language = Field(description="Programming language for execution")
    code: str = Field(max_length=1_000_000, description="Code to execute (max 1MB)")
    timeout: int = Field(ge=0, le=300, default=0, description="Execution timeout in seconds (0=no timeout, max 300s)")
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables (max 100, see BLOCKED_ENV_VARS in guest-agent)",
    )

    @field_validator("env_vars")
    @classmethod
    def validate_env_vars(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate environment variable names and values.

        Security: Rejects control characters to prevent terminal escape injection,
        log injection, and protocol manipulation attacks.
        """
        from exec_sandbox.constants import (  # noqa: PLC0415
            ENV_VAR_FORBIDDEN_CONTROL_CHARS,
            MAX_ENV_VAR_NAME_LENGTH,
            MAX_ENV_VAR_VALUE_LENGTH,
            MAX_ENV_VARS,
        )

        if len(v) > MAX_ENV_VARS:
            raise ValueError(f"Too many environment variables: {len(v)} (max {MAX_ENV_VARS})")

        for key, value in v.items():
            # Validate name length
            if not key or len(key) > MAX_ENV_VAR_NAME_LENGTH:
                raise ValueError(f"Invalid env var name length: {len(key)} (must be 1-{MAX_ENV_VAR_NAME_LENGTH})")

            # Validate value length
            if len(value) > MAX_ENV_VAR_VALUE_LENGTH:
                raise ValueError(
                    f"Env var '{key}' value too large: {len(value)} bytes (max {MAX_ENV_VAR_VALUE_LENGTH})"
                )

            # Check for control characters in name
            for char in key:
                if ord(char) in ENV_VAR_FORBIDDEN_CONTROL_CHARS:
                    raise ValueError(f"Env var name contains forbidden control character: 0x{ord(char):02X}")

            # Check for control characters in value
            for char in value:
                if ord(char) in ENV_VAR_FORBIDDEN_CONTROL_CHARS:
                    raise ValueError(f"Env var '{key}' value contains forbidden control character: 0x{ord(char):02X}")

        return v


class InstallPackagesRequest(GuestAgentRequest):
    """Install packages via pip (Python) or bun (JavaScript).

    Guest agent enforces:
    - Package count limit: 50
    - Package name length: 214 chars (PyPI limit)
    - Version specifier required: pandas==2.0.0, lodash@4.17.21
    - Path traversal protection: no /, .., \\
    - Timeout: 300s

    Response: Streaming messages (OutputChunkMessage for stdout/stderr, ExecutionCompleteMessage for completion).
    """

    action: Literal["install_packages"] = Field(default="install_packages")  # type: ignore[assignment]
    language: Language = Field(description="Programming language (python=pip/uv, javascript=bun, raw=shell)")
    packages: list[str] = Field(
        min_length=1,
        max_length=50,
        description="Package list with version specifiers (e.g., ['pandas==2.0.0', 'lodash@4.17.21'])",
    )
    timeout: int = Field(ge=0, le=300, default=300, description="Installation timeout in seconds (max 300s)")


# ============================================================================
# Streaming Response Models
# ============================================================================


class OutputChunkMessage(BaseModel):
    """Streaming output chunk from code execution.

    Batching strategy (Jan 2026 best practice):
    - Flush every 50ms (real-time feel, not 1s which is too slow)
    - OR every 64KB accumulated (prevent memory exhaustion)
    - OR on process completion (final flush)

    This allows real-time console debugging without memory issues on large outputs.
    Uses backpressure via bounded channel when buffer is full.
    """

    type: Literal["stdout", "stderr"] = Field(description="Output stream type")
    chunk: str = Field(
        max_length=10_000_000,
        description="Output chunk (batched over 50ms window, max 64KB per flush)",
    )


class ExecutionCompleteMessage(BaseModel):
    """Final completion message for ANY command (exec, install_packages, etc).

    Sent after stdout/stderr streaming completes. ALL commands stream output
    via OutputChunkMessage first, then send this minimal completion signal.
    """

    type: Literal["complete"] = "complete"
    exit_code: int = Field(description="Process exit code (0=success)")
    execution_time_ms: int = Field(description="Total execution time in milliseconds")
    spawn_ms: int | None = Field(
        default=None,
        description="Time for process spawn (fork/exec) in milliseconds (guest-reported)",
    )
    process_ms: int | None = Field(
        default=None,
        description="Time from spawn to process exit in milliseconds (guest-reported)",
    )


class StreamingErrorMessage(BaseModel):
    """Error message during streaming execution."""

    type: Literal["error"] = "error"
    message: str = Field(description="Error message")
    error_type: str = Field(description="Error classification")
    version: str | None = Field(default=None, description="Guest agent version")


class PongMessage(BaseModel):
    """Response to ping request."""

    type: Literal["pong"] = "pong"
    version: str = Field(description="Guest agent version")


# Discriminated union for streaming messages
StreamingMessage = Annotated[
    OutputChunkMessage | ExecutionCompleteMessage | PongMessage | StreamingErrorMessage,
    Field(discriminator="type"),
]

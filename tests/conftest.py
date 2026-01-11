"""Shared pytest fixtures for exec-sandbox tests."""

import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.scheduler import Scheduler

# ============================================================================
# Common Paths and Config Fixtures
# ============================================================================


@pytest.fixture
def images_dir() -> Path:
    """Path to built VM images directory."""
    return Path(__file__).parent.parent / "images" / "dist"


@pytest.fixture
def scheduler_config(images_dir: Path) -> SchedulerConfig:
    """SchedulerConfig with default test settings."""
    return SchedulerConfig(images_dir=images_dir)


@pytest.fixture
async def scheduler(scheduler_config: SchedulerConfig) -> AsyncGenerator[Scheduler, None]:
    """Scheduler instance for integration tests.

    Usage:
        async def test_something(scheduler: Scheduler) -> None:
            result = await scheduler.run(code="print(1)", language=Language.PYTHON)
    """
    async with Scheduler(scheduler_config) as sched:
        yield sched


# ============================================================================
# Test Utilities
# ============================================================================


async def async_iter(items):
    """Convert list to async iterator for testing."""
    for item in items:
        yield item


@pytest.fixture
def test_id():
    """Generate unique test ID."""
    return str(uuid.uuid4())


@pytest.fixture
def tenant_id():
    """Generate unique tenant ID."""
    return str(uuid.uuid4())


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.environment = "development"
    settings.max_concurrent_vms = 10
    settings.snapshot_cache_dir = Path("/tmp/snapshots")
    settings.base_images_dir = Path("/images")
    settings.kernel_path = Path("/images/kernels")
    settings.s3_bucket = None
    settings.s3_region = "us-east-1"
    return settings


@pytest.fixture
def mock_package_validator():
    """Create mock package validator."""
    validator = Mock()
    validator.validate = Mock(return_value=None)  # No-op for valid packages
    return validator


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["ENVIRONMENT"] = "development"
    os.environ["LOG_LEVEL"] = "DEBUG"
    yield
    # Cleanup
    os.environ.pop("ENVIRONMENT", None)
    os.environ.pop("LOG_LEVEL", None)


@pytest.fixture
def mock_vsock_connection():
    """Create mock vsock connection."""
    conn = AsyncMock()
    conn.send = AsyncMock()
    conn.receive = AsyncMock(return_value={"status": "success", "stdout": "test output"})
    conn.close = AsyncMock()
    return conn


# ============================================================================
# GuestChannel Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_guest_channel_success():
    """Mock GuestChannel with successful responses.

    Use for: happy path tests where execution succeeds.
    Includes spec=GuestChannel for type safety.
    """
    from exec_sandbox.guest_agent_protocol import ExecutionCompleteMessage, PongMessage
    from exec_sandbox.guest_channel import GuestChannel

    channel = AsyncMock(spec=GuestChannel)
    channel.connect = AsyncMock()
    # stream_messages returns async iterator
    channel.stream_messages = AsyncMock(
        return_value=async_iter(
            [
                ExecutionCompleteMessage(
                    exit_code=0,
                    execution_time_ms=42,
                )
            ]
        )
    )
    # send_request for ping operations
    channel.send_request = AsyncMock(return_value=PongMessage(version="1.0.0"))
    channel.close = AsyncMock()
    return channel


@pytest.fixture
def mock_guest_channel_error():
    """Mock GuestChannel that returns error responses.

    Use for: error path tests, validation failures.
    Includes spec=GuestChannel for type safety.
    """
    from exec_sandbox.guest_agent_protocol import PongMessage, StreamingErrorMessage
    from exec_sandbox.guest_channel import GuestChannel

    channel = AsyncMock(spec=GuestChannel)
    channel.connect = AsyncMock()
    # stream_messages returns async iterator with error
    channel.stream_messages = AsyncMock(
        return_value=async_iter(
            [
                StreamingErrorMessage(
                    message="Operation failed",
                    error_type="execution_error",
                    version="1.0.0",
                )
            ]
        )
    )
    # send_request for ping operations
    channel.send_request = AsyncMock(return_value=PongMessage(version="1.0.0"))
    channel.close = AsyncMock()
    return channel


@pytest.fixture
def make_mock_channel():
    """Factory for custom GuestChannel mock behavior.

    Args:
        response: StreamingMessage or list[StreamingMessage] to return
        connect_error: Exception to raise on connect()
        send_error: Exception to raise on send_request()
        stream_error: Exception to raise on stream_messages()

    Example:
        channel = make_mock_channel(
            response=[OutputChunkMessage(...), ExecutionCompleteMessage(...)],
            connect_error=TimeoutError("Connection timeout")
        )
    """
    from exec_sandbox.guest_agent_protocol import PongMessage
    from exec_sandbox.guest_channel import GuestChannel

    def _make(response=None, connect_error=None, send_error=None, stream_error=None):
        channel = AsyncMock(spec=GuestChannel)

        # Configure connect() behavior
        if connect_error:
            channel.connect = AsyncMock(side_effect=connect_error)
        else:
            channel.connect = AsyncMock()

        # Configure stream_messages() behavior
        if stream_error:
            channel.stream_messages = AsyncMock(side_effect=stream_error)
        elif response:
            # Support both single message and list of messages
            messages = response if isinstance(response, list) else [response]
            channel.stream_messages = AsyncMock(return_value=async_iter(messages))
        else:
            # Default: pong response
            channel.stream_messages = AsyncMock(return_value=async_iter([PongMessage(version="1.0.0")]))

        # Configure send_request() for backward compatibility (ping operations)
        if send_error:
            channel.send_request = AsyncMock(side_effect=send_error)
        else:
            # Default: pong response
            channel.send_request = AsyncMock(return_value=PongMessage(version="1.0.0"))

        channel.close = AsyncMock()
        return channel

    return _make


# ============================================================================
# QemuVM Mock Fixtures (for snapshot manager tests)
# ============================================================================


@pytest.fixture
def mock_vm_with_streaming():
    """Complete mock VM with streaming channel and async process.

    Use for: snapshot creation, package installation tests.
    Includes spec=QemuVM for type safety.
    """
    from exec_sandbox.guest_agent_protocol import ExecutionCompleteMessage
    from exec_sandbox.vm_manager import QemuVM

    vm = Mock(spec=QemuVM)
    vm.vm_id = "test-vm-123"
    vm.overlay_image = Path("/tmp/test-overlay.qcow2")

    # CRITICAL: Process must be AsyncMock for await vm.process.wait()
    vm.process = AsyncMock()
    vm.process.wait = AsyncMock(return_value=0)
    vm.process.pid = 12345
    vm.process.returncode = 0

    # Channel with streaming
    vm.channel = AsyncMock()
    vm.channel.connect = AsyncMock()
    vm.channel.close = AsyncMock()

    # Default: streaming install success
    async def default_stream(request, timeout):
        yield ExecutionCompleteMessage(exit_code=0, execution_time_ms=100)

    vm.channel.stream_messages = default_stream

    # Cleanup methods
    vm.destroy = AsyncMock()

    return vm


@pytest.fixture
def mock_vm_install_error():
    """VM that fails package installation."""
    from exec_sandbox.guest_agent_protocol import StreamingErrorMessage
    from exec_sandbox.vm_manager import QemuVM

    vm = Mock(spec=QemuVM)
    vm.vm_id = "test-vm-error"
    vm.overlay_image = Path("/tmp/test-overlay.qcow2")

    # CRITICAL: Process must be AsyncMock
    vm.process = AsyncMock()
    vm.process.wait = AsyncMock(return_value=1)
    vm.process.pid = 12346
    vm.process.returncode = 1

    vm.channel = AsyncMock()
    vm.channel.connect = AsyncMock()
    vm.channel.close = AsyncMock()

    async def error_stream(request, timeout):
        yield StreamingErrorMessage(message="Installation failed", error_type="execution_error", version="1.0.0")

    vm.channel.stream_messages = error_stream
    vm.destroy = AsyncMock()

    return vm

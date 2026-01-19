"""Shared pytest fixtures for exec-sandbox tests."""

import asyncio
import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_arch, detect_host_os
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_manager import VmManager, check_fast_balloon_available, check_hwaccel_available

# ============================================================================
# Shared Skip Markers
# ============================================================================

# Skip marker for timing-sensitive tests that require hardware acceleration.
# TCG (software emulation) is 10-50x slower than KVM/HVF, making these tests
# unreliable on GitHub Actions macOS runners (no nested virtualization).
skip_unless_hwaccel = pytest.mark.skipif(
    not check_hwaccel_available(),
    reason="Requires hardware acceleration (KVM/HVF) - TCG too slow for timing-sensitive tests",
)

# Skip marker for tests with tight timing assertions that include balloon overhead.
# Even with KVM available, nested virtualization (GitHub Actions runners on Azure)
# causes balloon operations to be 50-100x slower than bare-metal. This marker
# requires both hwaccel AND TSC_DEADLINE (x86_64) to ensure fast balloon ops.
# See check_fast_balloon_available() docstring for full rationale and references.
skip_unless_fast_balloon = pytest.mark.skipif(
    not check_fast_balloon_available(),
    reason=(
        "Requires fast balloon operations - nested virtualization (CI runners) causes "
        "balloon timeouts. TSC_DEADLINE CPU feature missing indicates degraded nested virt."
    ),
)

# Skip marker for Linux-only tests (cgroups, virtual memory ulimit, etc.)
skip_unless_linux = pytest.mark.skipif(
    detect_host_os() != HostOS.LINUX,
    reason="This test requires Linux (cgroups, virtual memory limits, etc.)",
)

# Skip marker for macOS-only tests (HVF, macOS-specific behavior, etc.)
skip_unless_macos = pytest.mark.skipif(
    detect_host_os() != HostOS.MACOS,
    reason="This test requires macOS",
)

# Skip marker for x86_64-only tests
skip_unless_x86_64 = pytest.mark.skipif(
    detect_host_arch() != HostArch.X86_64,
    reason="This test requires x86_64 architecture",
)

# Skip marker for ARM64-only tests
skip_unless_aarch64 = pytest.mark.skipif(
    detect_host_arch() != HostArch.AARCH64,
    reason="This test requires ARM64/aarch64 architecture",
)

# Combined markers for specific platform+arch combinations
skip_unless_macos_x86_64 = pytest.mark.skipif(
    not (detect_host_os() == HostOS.MACOS and detect_host_arch() == HostArch.X86_64),
    reason="This test requires macOS on Intel (x86_64)",
)

skip_unless_macos_arm64 = pytest.mark.skipif(
    not (detect_host_os() == HostOS.MACOS and detect_host_arch() == HostArch.AARCH64),
    reason="This test requires macOS on Apple Silicon (ARM64)",
)

# Skip marker for tests affected by Python 3.12 asyncio subprocess bug.
# Bug: asyncio.create_subprocess_exec() with piped output hangs indefinitely
# when tasks are cancelled during pipe connection phase.
# Fixed in Python 3.13+ via https://github.com/python/cpython/pull/140805
# See: https://github.com/python/cpython/issues/103847
skip_on_python_312_subprocess_bug = pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason=(
        "Skipped due to CPython bug #103847: asyncio subprocess hangs on task "
        "cancellation during pipe connection. Fixed in Python 3.13+. "
        "See: https://github.com/python/cpython/issues/103847"
    ),
)

# ============================================================================
# Common Paths and Config Fixtures
# ============================================================================


@pytest.fixture
def images_dir() -> Path:
    """Path to built VM images directory."""
    return Path(__file__).parent.parent / "images" / "dist"


@pytest.fixture
def scheduler_config(images_dir: Path) -> SchedulerConfig:
    """SchedulerConfig with default test settings.

    Uses pre-built images from images/dist/ directory.
    Disables auto_download_assets since images are provided locally.
    """
    return SchedulerConfig(images_dir=images_dir, auto_download_assets=False)


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
# VmManager Fixtures
# ============================================================================


@pytest.fixture
def vm_settings(images_dir: Path):
    """Settings for VM tests with hardware acceleration."""
    from exec_sandbox.settings import Settings

    return Settings(
        base_images_dir=images_dir,
        kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
        max_concurrent_vms=4,
    )


@pytest.fixture
async def vm_manager(vm_settings) -> AsyncGenerator[VmManager, None]:
    """VmManager with hardware acceleration (started).

    Automatically calls start() to start the overlay pool daemon,
    and stop() for cleanup.
    """
    async with VmManager(vm_settings) as manager:  # type: ignore[arg-type]
        yield manager


@pytest.fixture
def emulation_settings(images_dir: Path):
    """Settings with forced software emulation (no KVM/HVF)."""
    from exec_sandbox.settings import Settings

    return Settings(
        base_images_dir=images_dir,
        kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
        max_concurrent_vms=4,
        force_emulation=True,
    )


@pytest.fixture
async def emulation_vm_manager(emulation_settings) -> AsyncGenerator[VmManager, None]:
    """VmManager configured for software emulation (started).

    Automatically calls start() to start the overlay pool daemon,
    and stop() for cleanup.
    """
    async with VmManager(emulation_settings) as manager:  # type: ignore[arg-type]
        yield manager


@pytest.fixture
def unit_test_settings():
    """Settings for unit tests that don't need real images.

    Uses nonexistent paths since unit tests don't boot actual VMs.
    """
    from exec_sandbox.settings import Settings

    return Settings(
        base_images_dir=Path("/nonexistent"),
        kernel_path=Path("/nonexistent"),
        max_concurrent_vms=4,
    )


@pytest.fixture
def unit_test_vm_manager(unit_test_settings):
    """VmManager for unit tests that don't boot real VMs."""
    from exec_sandbox.vm_manager import VmManager

    return VmManager(unit_test_settings)  # type: ignore[arg-type]


# ============================================================================
# VmManager Fixture Factories (for tests needing custom Settings)
# ============================================================================


@pytest.fixture
def make_vm_settings(images_dir: Path):
    """Factory to create Settings with optional overrides.

    Usage:
        def test_something(make_vm_settings, tmp_path):
            settings = make_vm_settings(snapshot_cache_dir=tmp_path / "cache")
    """
    from typing import Any

    from exec_sandbox.settings import Settings

    def _make(**overrides: Any) -> Settings:
        defaults: dict[str, Any] = {
            "base_images_dir": images_dir,
            "kernel_path": images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            "max_concurrent_vms": 4,
        }
        defaults.update(overrides)
        return Settings(**defaults)

    return _make


@pytest.fixture
def make_vm_manager(make_vm_settings):  # type: ignore[no-untyped-def]
    """Factory to create VmManager with optional Settings overrides.

    Usage:
        def test_something(make_vm_manager, tmp_path):
            vm_manager = make_vm_manager(snapshot_cache_dir=tmp_path / "cache")
    """
    from typing import Any

    from exec_sandbox.vm_manager import VmManager

    def _make(**settings_overrides: Any) -> VmManager:
        settings = make_vm_settings(**settings_overrides)
        return VmManager(settings)  # type: ignore[arg-type]

    return _make


# ============================================================================
# Test Utilities
# ============================================================================


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
def assert_no_pending_tasks():
    """Fixture that fails if any tasks are pending after the test.

    Use this fixture in tests that create background tasks to ensure
    proper cleanup. Python 3.14 has stricter detection of orphaned tasks.

    Usage:
        async def test_something(assert_no_pending_tasks) -> None:
            # ... test code that creates tasks ...
            assert_no_pending_tasks()  # Call at end to verify cleanup
    """

    def _check() -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, nothing to check
            return

        pending = [t for t in asyncio.all_tasks(loop) if not t.done() and t is not asyncio.current_task(loop)]
        if pending:
            task_names = [t.get_name() for t in pending]
            pytest.fail(f"Pending tasks after test: {task_names}")

    return _check

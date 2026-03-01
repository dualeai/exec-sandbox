"""Shared pytest fixtures for exec-sandbox tests."""

import asyncio
import logging
import os
import random
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from exec_sandbox.config import SchedulerConfig
from exec_sandbox.platform_utils import HostArch, HostOS, detect_host_arch, detect_host_os
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.system_probes import check_fast_balloon_available, check_hwaccel_available
from exec_sandbox.vm_manager import VmManager

logger = logging.getLogger(__name__)

# ============================================================================
# QEMU Crash Retry (GitHub Actions CI only)
# ============================================================================
# GitHub Actions CI runners can trigger a transient QEMU crash:
#   Thread exhaustion: "qemu_thread_create: Resource temporarily unavailable"
#      → Host runs out of threads under parallel test load (EAGAIN)
#      → Causes SIGABRT (exit code -6). This is a host-side issue, not a test bug.
#
# Root cause: With TCG thread=single, each QEMU process uses ~5-8 threads at
# steady state (main loop, vCPU, call_rcu, iothread, signal handler). However,
# the I/O worker thread pool (QEMU util/thread-pool.c, aio=threads) can spike
# to 64 threads *per AioContext* during heavy disk I/O. When many VMs boot
# simultaneously (pytest -n auto with 4 workers), these transient spikes can
# exhaust OS thread limits. Measured locally: 5 threads (TCG minimal) to
# 8 threads (HVF full config); QEMU maintainer docs confirm ~4-7 baseline
# with unbounded I/O pool spikes.
#
# Note: The previous "regime_is_user: code should not be reached" crash (QEMU
# ARM64 TCG bug in versions < 9.0.4) is now caught early by a version check in
# qemu_cmd.py, which raises VmDependencyError before QEMU is launched. That
# crash was deterministic (not transient) and exited with code 0, not SIGABRT.
#
# We detect exact error strings and retry with backoff plus heavy jitter to
# desynchronize parallel workers on noisy CI hardware.
#
# Tuning rationale:
#   - 5 retries (not 3): thread exhaustion can persist through multiple boot
#     cycles when all workers retry in lockstep.
#   - Base 2 (not 3): base 3 grows too fast for 5 retries (3^4=81s, 3^5=243s).
#     Base 2 gives a gentler ramp: 2s, 4s, 8s, 16s, 32s.
#   - 10s jitter (not 5s): wider uniform spread better desynchronizes 4 parallel
#     workers, reducing the chance of simultaneous retry storms.
#   - Worst-case total: ~112s across 5 retries (acceptable vs 60-min CI timeout).

_QEMU_CRASH_MAX_RETRIES = 5
_QEMU_CRASH_BACKOFF_BASE_S = 2  # exponential: 2s, 4s, 8s, 16s, 32s
_QEMU_CRASH_JITTER_MAX_S = 10.0  # uniform random jitter [0, 10s)

# Exact substrings from QEMU's stderr/console that indicate retryable failures.
# Thread exhaustion: "Resource temporarily unavailable" / "SIGABRT"
# VM timeouts under TCG contention: VmBootTimeoutError (host watchdog) and
# VmTransientError (guest agent unresponsive). These are classified as
# "may succeed on retry" by the exception hierarchy (exceptions.py).
_QEMU_RETRYABLE_SIGNATURES = (
    "Resource temporarily unavailable",
    "SIGABRT",
    "VmBootTimeoutError",
    "VmTransientError",
)


def _is_retryable_qemu_crash(reports: list[pytest.TestReport]) -> bool:
    """Check if any test phase failed with a retryable QEMU crash signature."""
    for report in reports:
        if not report.failed:
            continue
        if not report.longrepr:
            continue
        text = str(report.longrepr)
        if any(sig in text for sig in _QEMU_RETRYABLE_SIGNATURES):
            return True
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> bool | None:
    """Retry tests on QEMU SIGABRT / thread exhaustion (GitHub Actions CI only).

    Uses exponential backoff (2^n) with full jitter ([0, 10s)) between retries
    to desynchronize parallel workers and avoid thundering herd on noisy CI.
    Properly tears down and re-creates fixtures on each attempt.
    """
    if not os.environ.get("GITHUB_ACTIONS"):
        return None  # Use default protocol outside CI

    from _pytest.runner import runtestprotocol

    for attempt in range(_QEMU_CRASH_MAX_RETRIES + 1):
        reports = runtestprotocol(item, nextitem=nextitem, log=False)

        if not _is_retryable_qemu_crash(reports) or attempt == _QEMU_CRASH_MAX_RETRIES:
            # Final attempt or non-crash failure — report as-is
            for report in reports:
                if attempt > 0:
                    report.sections.append(
                        (
                            "qemu-crash-retry",
                            f"Retried {attempt} time(s) due to transient QEMU crash",
                        )
                    )
                item.ihook.pytest_runtest_logreport(report=report)
            return True

        delay = _QEMU_CRASH_BACKOFF_BASE_S ** (attempt + 1) + random.uniform(0, _QEMU_CRASH_JITTER_MAX_S)
        logger.warning(
            "Transient QEMU crash on %s, retrying in %.1fs (attempt %d/%d)",
            item.nodeid,
            delay,
            attempt + 1,
            _QEMU_CRASH_MAX_RETRIES,
        )
        time.sleep(delay)

    return True  # pragma: no cover


# ============================================================================
# Shared Skip Markers
# ============================================================================

# Skip marker for timing-sensitive tests that require hardware acceleration.
# TCG (software emulation) is ~5-8x slower than KVM/HVF, making these tests
# unreliable on GitHub Actions macOS runners (no nested virtualization).
skip_unless_hwaccel = pytest.mark.skipif(
    not asyncio.run(check_hwaccel_available()),
    reason="Requires hardware acceleration (KVM/HVF) - TCG too slow for timing-sensitive tests",
)

# Skip marker for tests with tight timing assertions that include balloon overhead.
# Even with KVM available, nested virtualization (GitHub Actions runners on Azure)
# causes balloon operations to be 50-100x slower than bare-metal. This marker
# requires both hwaccel AND TSC_DEADLINE (x86_64) to ensure fast balloon ops.
# See check_fast_balloon_available() docstring for full rationale and references.
skip_unless_fast_balloon = pytest.mark.skipif(
    not asyncio.run(check_fast_balloon_available()),
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

    Timeout is 120s (vs library default 30s) to absorb parallel-load
    slowdowns: ``pytest -n auto`` runs ~10 QEMU VMs concurrently, and TCG
    emulation + L1 restore contention can inflate wall-clock times 4-5x.
    Individual tests should NOT override ``timeout_seconds`` unless they
    intentionally need a *shorter* deadline (e.g. testing the timeout
    mechanism itself).
    """
    return SchedulerConfig(images_dir=images_dir, auto_download_assets=False, default_timeout_seconds=120)


@pytest.fixture
async def scheduler(scheduler_config: SchedulerConfig) -> AsyncGenerator[Scheduler, None]:
    """Scheduler instance for integration tests.

    Usage:
        async def test_something(scheduler: Scheduler) -> None:
            result = await scheduler.run(code="print(1)", language=Language.PYTHON)
    """
    async with Scheduler(scheduler_config) as sched:
        yield sched


@pytest.fixture(params=["hwaccel", "emulation"])
async def dual_scheduler(
    request: pytest.FixtureRequest,
    images_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[Scheduler, None]:
    """Scheduler that runs each test under both hardware acceleration and TCG emulation.

    - hwaccel: Uses KVM/HVF if available, skips if not.
    - emulation: Forces QEMU TCG software emulation via EXEC_SANDBOX_FORCE_EMULATION.

    This ensures security properties hold regardless of QEMU backend.

    Uses the same 120s default timeout as ``scheduler_config`` for hwaccel.
    Emulation gets 240s because parallel TCG VMs on CI runners can starve
    each other — trivial code that takes <1s locally consumed the full 120s
    execution budget on a loaded macOS arm64 runner (cold-boot fallback +
    TCG + ``pytest -n auto`` contention).
    """
    if request.param == "hwaccel":
        if not await check_hwaccel_available():
            pytest.skip("Hardware acceleration not available")
        monkeypatch.delenv("EXEC_SANDBOX_FORCE_EMULATION", raising=False)
        timeout = 120
    else:
        monkeypatch.setenv("EXEC_SANDBOX_FORCE_EMULATION", "true")
        timeout = 240

    config = SchedulerConfig(images_dir=images_dir, auto_download_assets=False, default_timeout_seconds=timeout)
    async with Scheduler(config) as sched:
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
            settings = make_vm_settings(disk_snapshot_cache_dir=tmp_path / "cache")
    """
    from typing import Any

    from exec_sandbox.settings import Settings

    def _make(**overrides: Any) -> Settings:
        defaults: dict[str, Any] = {
            "base_images_dir": images_dir,
            "kernel_path": images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
        }
        defaults.update(overrides)
        return Settings(**defaults)

    return _make


@pytest.fixture
async def make_vm_manager(make_vm_settings):  # type: ignore[no-untyped-def]
    """Factory to create started VmManager with optional Settings overrides.

    Each VmManager is automatically started (overlay pool daemon, system probes)
    and stopped on fixture teardown via AsyncExitStack.

    Usage:
        async def test_something(make_vm_manager, tmp_path):
            vm_manager = await make_vm_manager(disk_snapshot_cache_dir=tmp_path / "cache")
    """
    from contextlib import AsyncExitStack
    from typing import Any

    from exec_sandbox.vm_manager import VmManager

    async with AsyncExitStack() as stack:

        async def _make(**settings_overrides: Any) -> VmManager:
            settings = make_vm_settings(**settings_overrides)
            manager = VmManager(settings)  # type: ignore[arg-type]
            await stack.enter_async_context(manager)
            return manager

        yield _make


# ============================================================================
# Network test constants — shared by test_dns_filtering / test_ech_proxy_behavior
# ============================================================================
# These are NOT fixtures because they are used at module-level (f-string
# interpolation into code-generation helpers), not inside test functions.

NET_CONNECT_TIMEOUT_S = 5  # TCP connect timeout (socket, curl --connect-timeout)
NET_OP_TIMEOUT_S = 10  # Full operation timeout (AbortSignal, urllib, curl --max-time)
NET_SAFETY_TIMEOUT_S = 15  # Process-level kill — catches musl DNS hang under TCG
NET_RETRY_COUNT = 2  # Retries for allowed-domain tests (3 total attempts)
NET_RETRY_BACKOFF_S = 1  # Linear backoff base (1s, 2s between retries)

# Safety preambles — kill the process if musl DNS blocks beyond the safety timeout.
# threading.Timer fires even when the main thread is stuck in getaddrinfo().
PY_SAFETY = f"""\nimport threading, os\nthreading.Timer({NET_SAFETY_TIMEOUT_S}, lambda: (print("BLOCKED:Timeout", flush=True), os._exit(1))).start()\n"""
# setTimeout fires even when Bun's main thread is waiting on DNS (thread-pool).
JS_SAFETY = (
    f'setTimeout(() => {{ console.log("BLOCKED:Timeout"); process.exit(1); }}, {NET_SAFETY_TIMEOUT_S * 1000});\n'
)


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

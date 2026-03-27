"""Performance benchmarks for exec-sandbox.

Benchmarks real VM operations: cold boot, warm pool, session exec round-trip,
network overhead, and memory configurations.  Also benchmarks system probes
that run on every Scheduler init (subprocess cost, cached per process).

Parameterized by Language enum so new languages are automatically covered.

Requires: QEMU, VM images (from `make build` or CI artifacts), KVM (or TCG fallback).

Run with: uv run pytest tests/benchmarks/ --codspeed -v --no-cov -p no:xdist -o "addopts="
"""

from __future__ import annotations

from pathlib import Path

import pytest

from exec_sandbox import Scheduler, SchedulerConfig
from exec_sandbox.constants import DEFAULT_MEMORY_MB, MIN_MEMORY_MB
from exec_sandbox.models import Language
from exec_sandbox.system_probes import (
    check_tsc_deadline,
    detect_accel_type,
    probe_cache,
    probe_io_uring_support,
    probe_qemu_sandbox_support,
    probe_qemu_version,
    probe_unshare_support,
)

pytestmark = pytest.mark.benchmark

# Timeout generous enough for TCG fallback on CI
_TIMEOUT = 120

# Minimal code per language — just enough to prove the REPL works.
# Keyed by every Language variant; the assert below catches missing entries.
_HELLO_CODE: dict[Language, str] = {
    Language.PYTHON: "print('ok')",
    Language.JAVASCRIPT: "console.log('ok')",
    Language.RAW: "echo ok",
}
assert set(_HELLO_CODE) == set(Language), f"_HELLO_CODE missing languages: {set(Language) - set(_HELLO_CODE)}"

# Derive param list from the Language enum so new variants are auto-covered
_LANGUAGE_PARAMS = [pytest.param(lang, id=lang.value) for lang in Language]

# Pre-built images from images/dist/ — same as conftest.py scheduler_config.
# CI builds artifacts from current commit; locally requires `make build`.
_IMAGES_DIR = Path(__file__).parent.parent.parent / "images" / "dist"
_CONFIG = SchedulerConfig(
    images_dir=_IMAGES_DIR,
    auto_download_assets=False,
    default_timeout_seconds=_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Cold boot — full VM lifecycle, parameterized by language
# Measures: Scheduler init + admission + overlay + QEMU fork + kernel boot
# + guest-agent + REPL startup + code exec + teardown.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", _LANGUAGE_PARAMS)
async def test_bench_cold_boot(language: Language) -> None:
    """Cold boot + execute — full VM lifecycle per language."""
    async with Scheduler(_CONFIG) as scheduler:
        result = await scheduler.run(code=_HELLO_CODE[language], language=language)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Cold boot with network — measures gvproxy overhead
# Delta vs cold_boot isolates: gvproxy fork + socket + DNS filter init.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", _LANGUAGE_PARAMS)
async def test_bench_cold_boot_network(language: Language) -> None:
    """Cold boot with network enabled — measures gvproxy overhead."""
    async with Scheduler(_CONFIG) as scheduler:
        result = await scheduler.run(
            code=_HELLO_CODE[language],
            language=language,
            allow_network=True,
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Memory configurations — different VM sizes affect boot time and memory mapping
# MIN (128) is the floor, DEFAULT (192) is standard, 512 is a typical production
# size for package-heavy workloads (pandas, torch, etc.).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "memory_mb",
    [
        pytest.param(MIN_MEMORY_MB, id=f"{MIN_MEMORY_MB}mb"),
        pytest.param(DEFAULT_MEMORY_MB, id=f"{DEFAULT_MEMORY_MB}mb"),
        pytest.param(512, id="512mb"),
    ],
)
async def test_bench_cold_boot_memory(memory_mb: int) -> None:
    """Cold boot Python at different memory sizes — measures memory scaling."""
    async with Scheduler(_CONFIG) as scheduler:
        result = await scheduler.run(
            code=_HELLO_CODE[Language.PYTHON],
            language=Language.PYTHON,
            memory_mb=memory_mb,
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Warm pool — pool fill + warm hit, parameterized by language
# Covers: background VM boot, pool management, warm VM acquisition.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", _LANGUAGE_PARAMS)
async def test_bench_warm_pool(language: Language) -> None:
    """Warm pool: fill 1 VM + serve one request from pool."""
    async with Scheduler(_CONFIG.model_copy(update={"warm_pool_size": 1})) as scheduler:
        await scheduler.wait_pool_ready()
        result = await scheduler.run(code=_HELLO_CODE[language], language=language)
        assert result.exit_code == 0
        assert result.warm_pool_hit


# ---------------------------------------------------------------------------
# Session exec — boot once, measure REPL round-trip
# 10 exec() calls dominate over the single boot cost, so regressions in
# virtio-serial protocol, guest-agent REPL, or streaming parsing show up.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", _LANGUAGE_PARAMS)
async def test_bench_session_exec(language: Language) -> None:
    """Session: 1 boot + 10 exec() calls — isolates REPL round-trip."""
    async with Scheduler(_CONFIG) as scheduler:
        async with await scheduler.session(language=language) as session:
            for _i in range(10):
                result = await session.exec(_HELLO_CODE[language])
                assert result.exit_code == 0


# ---------------------------------------------------------------------------
# System probes — subprocess cost paid on every Scheduler init.
# Each probe spawns a QEMU subprocess (or syscall test) and caches the result.
# Regressions here directly impact cold-start latency for the first VM.
# Cache is reset before each call to measure real (uncached) cost.
# ---------------------------------------------------------------------------


async def test_bench_probe_qemu_version() -> None:
    """Probe: detect QEMU binary version via --version."""
    probe_cache.reset("qemu_version")
    result = await probe_qemu_version()
    assert result is not None


async def test_bench_probe_qemu_sandbox() -> None:
    """Probe: detect -sandbox (seccomp) support via trial execution."""
    probe_cache.reset("qemu_sandbox")
    await probe_qemu_sandbox_support()


async def test_bench_probe_accel_type() -> None:
    """Probe: detect virtualization mode (KVM/HVF/TCG) via -accel help + OS checks."""
    probe_cache.reset("kvm")
    probe_cache.reset("hvf")
    probe_cache.reset("qemu_accels")
    result = await detect_accel_type()
    assert result is not None


async def test_bench_probe_io_uring() -> None:
    """Probe: detect io_uring support via syscall test."""
    probe_cache.reset("io_uring")
    await probe_io_uring_support()


async def test_bench_probe_unshare() -> None:
    """Probe: detect unshare (namespace) support."""
    probe_cache.reset("unshare")
    await probe_unshare_support()


async def test_bench_probe_tsc_deadline() -> None:
    """Probe: detect TSC deadline timer support."""
    probe_cache.reset("tsc_deadline")
    await check_tsc_deadline()

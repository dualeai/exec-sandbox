"""Background resource monitor for VM observability.

Reads live VM resource usage (cgroup or psutil fallback) on a periodic tick.
Stateless - reads current values each tick, no history accumulation.

Data sources:
- Primary (Linux): cgroup memory.current + cpu.stat (zero guest trust surface)
- Fallback (macOS): psutil Process.memory_info().rss
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from exec_sandbox._logging import get_logger
from exec_sandbox.cgroup import is_cgroup_available, read_cgroup_current
from exec_sandbox.constants import RESOURCE_MONITOR_INTERVAL_SECONDS
from exec_sandbox.platform_utils import HostOS, detect_host_os

if TYPE_CHECKING:
    from exec_sandbox.admission import ResourceAdmissionController
    from exec_sandbox.vm_manager import VmManager

logger = get_logger(__name__)


class ResourceMonitor:
    """Background task reading live VM resource usage for observability.

    NOT used for admission decisions - purely for logging/debugging.
    Reads cgroup stats on Linux, falls back to psutil RSS on macOS.
    """

    def __init__(
        self,
        vm_manager: VmManager,
        admission: ResourceAdmissionController,
        interval_seconds: float = RESOURCE_MONITOR_INTERVAL_SECONDS,
    ) -> None:
        self._vm_manager = vm_manager
        self._admission = admission
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._host_os = detect_host_os()

    async def start(self) -> None:
        """Start the background monitoring task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "Resource monitor started",
            extra={"interval_seconds": self._interval},
        )

    async def stop(self) -> None:
        """Stop the background monitoring task."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info("Resource monitor stopped")

    async def _monitor_loop(self) -> None:
        """Periodic monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._tick()
            except asyncio.CancelledError:
                break
            except (OSError, RuntimeError, ValueError):
                logger.debug("Resource monitor tick failed", exc_info=True)

    async def _tick(self) -> None:
        """Single monitoring tick - read and log resource usage."""
        active_vms = self._vm_manager.get_active_vms()
        if not active_vms:
            return

        # Collect per-VM stats
        total_memory_mb = 0
        total_cpu_ms = 0

        for vm in active_vms.values():
            cpu_ms: int | None = None
            memory_mb: int | None = None

            # Try cgroup first (Linux)
            if is_cgroup_available(vm.cgroup_path):
                cpu_ms, memory_mb = await read_cgroup_current(vm.cgroup_path)

            # Fallback to psutil RSS (macOS)
            if memory_mb is None and self._host_os == HostOS.MACOS:
                memory_mb = await self._read_psutil_rss(vm.process.pid)

            if memory_mb is not None:
                total_memory_mb += memory_mb
            if cpu_ms is not None:
                total_cpu_ms += cpu_ms

        # Log aggregate stats alongside admission snapshot
        snap = self._admission.snapshot()
        logger.debug(
            "Resource monitor tick",
            extra={
                "active_vms": len(active_vms),
                "observed_memory_mb": total_memory_mb,
                "observed_cpu_ms": total_cpu_ms,
                "admission_allocated_memory_mb": round(snap.allocated_memory_mb),
                "admission_allocated_cpu": round(snap.allocated_cpu, 1),
                "admission_vm_slots": snap.allocated_vm_slots,
            },
        )

    @staticmethod
    async def _read_psutil_rss(pid: int | None) -> int | None:
        """Read process RSS via psutil (macOS fallback).

        Args:
            pid: Process ID to read

        Returns:
            RSS in MB, or None if read fails
        """
        if pid is None:
            return None
        try:
            import psutil  # noqa: PLC0415
        except ImportError:
            return None
        try:
            loop = asyncio.get_running_loop()
            proc = psutil.Process(pid)
            mem_info = await loop.run_in_executor(None, proc.memory_info)
            return mem_info.rss // (1024 * 1024)
        except (OSError, psutil.NoSuchProcess, psutil.AccessDenied):
            return None

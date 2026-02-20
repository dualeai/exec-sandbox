"""Scheduler - main entry point for exec-sandbox.

The Scheduler provides a simple API for executing code in isolated microVMs.
Inspired by aiojobs - explicit resource management via async context manager.

Architecture:
- VM lifecycle: VMs are NEVER reused. Each run() gets a fresh VM, destroyed after.
- Warm pool: Pre-started VMs waiting for commands (faster than cold boot).
- Snapshot cache: L2 (local disk cache) + L3 (S3) for package installation speedup.
- Backpressure: resource-aware admission control prevents OOM from unbounded VM creation.

Example:
    ```python
    from exec_sandbox import Scheduler, SchedulerConfig

    # Basic usage
    async with Scheduler() as scheduler:
        result = await scheduler.run(
            code="print('hello')",
            language="python",
        )
        print(result.stdout)  # "hello\\n"

    # With packages and streaming
    async with Scheduler() as scheduler:
        result = await scheduler.run(
            code="import pandas; print(pandas.__version__)",
            language="python",
            packages=["pandas==2.2.0"],
            on_stdout=lambda chunk: print(chunk, end=""),
        )

    # Production config with S3 cache
    config = SchedulerConfig(
        s3_bucket="my-snapshots",
    )
    async with Scheduler(config) as scheduler:
        result = await scheduler.run(code="...", language="python")
    ```
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable  # noqa: TC003 - Used at runtime for on_stdout/on_stderr parameters
from pathlib import Path  # noqa: TC003 - Used at runtime
from typing import TYPE_CHECKING, Self

from exec_sandbox import constants
from exec_sandbox._logging import get_logger
from exec_sandbox.config import SchedulerConfig
from exec_sandbox.exceptions import SandboxError, VmConfigError
from exec_sandbox.models import ExecutionResult, ExposedPort, Language, PortMapping, TimingBreakdown
from exec_sandbox.port_forward import resolve_port_mappings
from exec_sandbox.session import Session
from exec_sandbox.settings import Settings

if TYPE_CHECKING:
    from exec_sandbox.qemu_vm import QemuVM
    from exec_sandbox.resource_monitor import ResourceMonitor
    from exec_sandbox.snapshot_manager import SnapshotManager
    from exec_sandbox.vm_manager import VmManager
    from exec_sandbox.warm_vm_pool import WarmVMPool

logger = get_logger(__name__)


class Scheduler:
    """Manages microVM lifecycle and code execution.

    The Scheduler is the main entry point for exec-sandbox. It handles:
    - VM pool management (max concurrent VMs, backpressure)
    - Warm VM pool (pre-booted VMs for instant execution)
    - Snapshot caching (L2 local + L3 S3)
    - Package validation

    Thread-safety: Single Scheduler per process. Use asyncio for concurrency.

    Lifecycle:
        async with Scheduler(config) as scheduler:
            # Scheduler is ready
            result = await scheduler.run(...)
        # All VMs destroyed, resources cleaned up

    Attributes:
        config: SchedulerConfig (immutable)
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        """Initialize Scheduler with configuration.

        Args:
            config: Optional SchedulerConfig. Uses defaults if None.

        Note:
            The Scheduler is NOT ready for use after __init__.
            Use as async context manager: `async with Scheduler() as s:`
        """
        self.config = config or SchedulerConfig()
        self._images_dir: Path | None = None
        self._settings: Settings | None = None
        self._vm_manager: VmManager | None = None
        self._snapshot_manager: SnapshotManager | None = None
        self._warm_pool: WarmVMPool | None = None
        self._resource_monitor: ResourceMonitor | None = None
        self._started = False

    async def __aenter__(self) -> Self:
        """Start scheduler and initialize resources.

        Startup sequence:
        1. Resolve images directory (downloads from GitHub if needed)
        2. Create Settings from SchedulerConfig
        3. Initialize VmManager
        4. Initialize SnapshotManager (if S3 configured)
        5. Start WarmVMPool (if warm_pool_size > 0)

        Returns:
            Self for use in context

        Raises:
            SandboxError: Startup failed
            FileNotFoundError: Assets not found and auto_download_assets=False
            AssetDownloadError: Failed to download required assets
        """
        if self._started:
            raise SandboxError("Scheduler already started")

        logger.info(
            "Starting scheduler",
            extra={
                "warm_pool_size": self.config.warm_pool_size,
                "s3_enabled": self.config.s3_bucket is not None,
                "auto_download_assets": self.config.auto_download_assets,
            },
        )

        # Resolve images directory (downloads if needed and allowed)
        from exec_sandbox.assets import ensure_assets  # noqa: PLC0415

        self._images_dir = await ensure_assets(
            override=self.config.images_dir,
            download=self.config.auto_download_assets,
        )

        # Create Settings from SchedulerConfig
        self._settings = self._create_settings()

        # Initialize VmManager (handles backpressure internally via semaphore)
        from exec_sandbox.vm_manager import VmManager  # noqa: PLC0415

        self._vm_manager = VmManager(self._settings)
        await self._vm_manager.start()  # Pre-warms all system probe caches

        # Initialize SnapshotManager (L2 local cache always available, L3 S3 optional)
        from exec_sandbox.snapshot_manager import SnapshotManager  # noqa: PLC0415

        self._snapshot_manager = SnapshotManager(self._settings, self._vm_manager)

        # Initialize WarmVMPool (optional)
        if self.config.warm_pool_size > 0:
            from exec_sandbox.warm_vm_pool import WarmVMPool  # noqa: PLC0415

            self._warm_pool = WarmVMPool(self._vm_manager, self.config, self._snapshot_manager)
            await self._warm_pool.start()

        # Start ResourceMonitor (always-on, core component for observability)
        from exec_sandbox.resource_monitor import ResourceMonitor  # noqa: PLC0415

        self._resource_monitor = ResourceMonitor(
            vm_manager=self._vm_manager,
            admission=self._vm_manager.admission,
            interval_seconds=self.config.resource_monitor_interval_seconds,
        )
        await self._resource_monitor.start()

        self._started = True
        logger.info("Scheduler started successfully")
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc_val: BaseException | None, _exc_tb: object
    ) -> None:
        """Shutdown scheduler and clean up all resources.

        Shutdown sequence:
        1. Stop WarmVMPool (drains and destroys pre-booted VMs)
        2. Destroy any remaining VMs
        3. Close SnapshotManager

        Always completes cleanup, even on exceptions.
        """
        logger.info("Shutting down scheduler")

        # Stop ResourceMonitor first
        if self._resource_monitor:
            try:
                await self._resource_monitor.stop()
            except (OSError, RuntimeError) as e:
                logger.error("ResourceMonitor stop error", extra={"error": str(e)})

        # Stop WarmVMPool first (drains VMs)
        if self._warm_pool:
            try:
                await self._warm_pool.stop()
            except (OSError, RuntimeError, TimeoutError) as e:
                logger.error("WarmVMPool stop error", extra={"error": str(e)})

        # Destroy any remaining VMs
        if self._vm_manager:
            active_vms = self._vm_manager.get_active_vms()
            if active_vms:
                logger.warning(
                    "Destroying remaining VMs on shutdown",
                    extra={"count": len(active_vms)},
                )
                destroy_tasks = [self._vm_manager.destroy_vm(vm) for vm in active_vms.values()]
                await asyncio.gather(*destroy_tasks, return_exceptions=True)

            # Stop VmManager (includes overlay pool cleanup)
            await self._vm_manager.stop()

        self._started = False
        logger.info("Scheduler shutdown complete")

    async def run(
        self,
        code: str,
        *,
        language: str | Language,
        packages: list[str] | None = None,
        timeout_seconds: int | None = None,
        memory_mb: int | None = None,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        expose_ports: list[PortMapping | int] | None = None,
        env_vars: dict[str, str] | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Execute code in an isolated microVM.

        Each call gets a fresh VM that is destroyed after execution.
        VMs are NEVER reused between runs (security guarantee).

        Args:
            code: Source code to execute.
            language: Programming language ("python", "javascript", or "raw").
            packages: Optional list of packages to install (e.g., ["pandas==2.2.0"]).
                Validated against allowlist if enable_package_validation=True.
            timeout_seconds: Execution timeout. Default: config.default_timeout_seconds.
            memory_mb: Guest VM memory in MB. Default: config.default_memory_mb.
            allow_network: Enable network access for the VM. Default: False.
            allowed_domains: Whitelist of domains if allow_network=True.
                If None/empty and allow_network=True, all domains allowed.
            expose_ports: List of ports to expose from guest to host.
                Can be PortMapping objects or integers (shorthand for internal port).
                Works independently of allow_network:
                - Mode 1 (allow_network=False): Port forwarding via QEMU hostfwd (no internet)
                - Mode 2 (allow_network=True): Port forwarding via gvproxy API (with internet)
            env_vars: Environment variables to set in the VM.
            on_stdout: Callback for stdout chunks (streaming). Called as chunks arrive.
            on_stderr: Callback for stderr chunks (streaming). Called as chunks arrive.

        Returns:
            ExecutionResult with stdout, stderr, exit_code, timing info, and exposed_ports.

        Raises:
            SandboxError: Scheduler not started.
            PackageNotAllowedError: Package not in allowlist.
            VmError: VM creation or execution failed.
            VmTimeoutError: Execution exceeded timeout.
            ValueError: Too many exposed ports or invalid port numbers.

        Example:
            ```python
            # Basic execution
            result = await scheduler.run("print(1+1)", language="python")
            assert result.stdout.strip() == "2"

            # With packages
            result = await scheduler.run(
                "import numpy; print(numpy.__version__)",
                language="python",
                packages=["numpy==1.26.0"],
            )

            # With port forwarding (no internet)
            result = await scheduler.run(
                "python -m http.server 8080",
                language="python",
                expose_ports=[PortMapping(internal=8080, external=3000)],
            )
            print(result.exposed_ports[0].url)  # http://127.0.0.1:3000

            # With streaming
            result = await scheduler.run(
                "for i in range(5): print(i)",
                language="python",
                on_stdout=lambda chunk: print(f"[OUT] {chunk}"),
            )
            ```
        """
        # Validate and apply defaults
        if timeout_seconds is not None and (timeout_seconds < 1 or timeout_seconds > constants.MAX_TIMEOUT_SECONDS):
            raise ValueError(
                f"timeout_seconds must be between 1 and {constants.MAX_TIMEOUT_SECONDS}, got {timeout_seconds}"
            )
        timeout = timeout_seconds if timeout_seconds is not None else self.config.default_timeout_seconds

        vm, resolved_ports, is_cold_boot = await self._prepare_vm(
            language=language,
            packages=packages,
            memory_mb=memory_mb,
            allow_network=allow_network,
            allowed_domains=allowed_domains,
            expose_ports=expose_ports,
            task_id=f"run-{id(code)}",
        )

        run_start_time = asyncio.get_event_loop().time()
        try:
            # Execute code
            execute_start_time = asyncio.get_event_loop().time()
            result = await vm.execute(
                code=code,
                timeout_seconds=timeout,
                env_vars=env_vars,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
            execute_end_time = asyncio.get_event_loop().time()

            # Calculate timing
            execute_ms = round((execute_end_time - execute_start_time) * 1000)
            total_ms = round((execute_end_time - run_start_time) * 1000)

            # For warm pool: setup/boot are "free" (happened at service startup)
            # For cold boot: use actual setup/boot times from VM
            setup_ms = vm.setup_ms if is_cold_boot and vm.setup_ms is not None else 0
            boot_ms = vm.boot_ms if is_cold_boot and vm.boot_ms is not None else 0
            # Granular setup timing
            overlay_ms = vm.overlay_ms if is_cold_boot and vm.overlay_ms is not None else 0
            # Granular boot timing
            qemu_cmd_build_ms = vm.qemu_cmd_build_ms if is_cold_boot and vm.qemu_cmd_build_ms is not None else 0
            gvproxy_start_ms = vm.gvproxy_start_ms if is_cold_boot and vm.gvproxy_start_ms is not None else 0
            qemu_fork_ms = vm.qemu_fork_ms if is_cold_boot and vm.qemu_fork_ms is not None else 0
            guest_wait_ms = vm.guest_wait_ms if is_cold_boot and vm.guest_wait_ms is not None else 0
            # Retry tracking (0 for warm pool since no boot retry needed)
            boot_retries = vm.timing.boot_retries if is_cold_boot and vm.timing.boot_retries is not None else 0

            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                execution_time_ms=result.execution_time_ms,
                external_cpu_time_ms=result.external_cpu_time_ms,
                external_memory_peak_mb=result.external_memory_peak_mb,
                timing=TimingBreakdown(
                    setup_ms=setup_ms,
                    boot_ms=boot_ms,
                    execute_ms=execute_ms,
                    total_ms=total_ms,
                    connect_ms=result.timing.connect_ms,
                    overlay_ms=overlay_ms,
                    qemu_cmd_build_ms=qemu_cmd_build_ms,
                    gvproxy_start_ms=gvproxy_start_ms,
                    qemu_fork_ms=qemu_fork_ms,
                    guest_wait_ms=guest_wait_ms,
                    boot_retries=boot_retries,
                ),
                warm_pool_hit=not is_cold_boot,
                spawn_ms=result.spawn_ms,  # Pass through from guest
                process_ms=result.process_ms,  # Pass through from guest
                exposed_ports=resolved_ports,  # Port forwarding mappings
            )

        finally:
            # Always destroy VM (never reused)
            # _vm_manager guaranteed non-None by _prepare_vm
            if self._vm_manager is not None:
                await self._vm_manager.destroy_vm(vm)

    async def session(
        self,
        *,
        language: str | Language,
        packages: list[str] | None = None,
        memory_mb: int | None = None,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        expose_ports: list[PortMapping | int] | None = None,
        idle_timeout_seconds: int | None = None,
    ) -> Session:
        """Create a persistent session for multi-step code execution.

        Returns a Session that keeps the VM alive across multiple exec() calls.
        State (variables, imports, functions) persists between executions.

        The Session owns the VM lifecycle - the VM is destroyed when the
        session is closed (explicitly, via context manager, or by idle timeout).

        Args:
            language: Programming language ("python", "javascript", or "raw").
            packages: Optional list of packages to install (e.g., ["pandas==2.2.0"]).
            memory_mb: Guest VM memory in MB. Default: config.default_memory_mb.
            allow_network: Enable network access for the VM. Default: False.
            allowed_domains: Whitelist of domains if allow_network=True.
            expose_ports: List of ports to expose from guest to host.
            idle_timeout_seconds: Auto-close after inactivity. Default: config setting.

        Returns:
            Session for executing code. Use as async context manager:
                async with await scheduler.session(language="python") as sess:
                    await sess.exec("x = 42")
                    result = await sess.exec("print(x)")

        Raises:
            SandboxError: Scheduler not started.
            PackageNotAllowedError: Package not in allowlist.
            VmError: VM creation failed.

        Example:
            ```python
            async with Scheduler() as scheduler:
                async with await scheduler.session(language="python") as session:
                    await session.exec("x = 42")
                    result = await session.exec("print(x)")
                    assert result.stdout.strip() == "42"
                    print(f"exec_count={session.exec_count}")
            ```
        """
        idle_timeout = idle_timeout_seconds or self.config.session_idle_timeout_seconds

        vm, _resolved_ports, _is_cold_boot = await self._prepare_vm(
            language=language,
            packages=packages,
            memory_mb=memory_mb,
            allow_network=allow_network,
            allowed_domains=allowed_domains,
            expose_ports=expose_ports,
            task_id=f"session-{id(self)}",
        )

        # _vm_manager guaranteed non-None by _prepare_vm
        if self._vm_manager is None:
            raise SandboxError("Scheduler resources not initialized")

        logger.info(
            "Session created",
            extra={
                "vm_id": vm.vm_id,
                "language": language,
                "idle_timeout_seconds": idle_timeout,
            },
        )

        return Session(
            vm=vm,
            vm_manager=self._vm_manager,
            idle_timeout_seconds=idle_timeout,
            default_timeout_seconds=self.config.default_timeout_seconds,
        )

    async def _prepare_vm(
        self,
        *,
        language: str | Language,
        packages: list[str] | None,
        memory_mb: int | None,
        allow_network: bool,
        allowed_domains: list[str] | None,
        expose_ports: list[PortMapping | int] | None,
        task_id: str,
    ) -> tuple[QemuVM, list[ExposedPort], bool]:
        """Create and boot a VM with the given configuration.

        Shared by run() and session() - handles validation, snapshots,
        warm pool, and cold boot. Returns the VM ready for execution.

        Args:
            language: Programming language.
            packages: Packages to install.
            memory_mb: Guest VM memory in MB (None = use default).
            allow_network: Enable network access.
            allowed_domains: Whitelist of domains.
            expose_ports: Ports to expose from guest to host.
            task_id: Identifier for this VM (e.g., "run-..." or "session-...").

        Returns:
            Tuple of (vm, resolved_ports, is_cold_boot).

        Raises:
            SandboxError: Scheduler not started.
            PackageNotAllowedError: Package not in allowlist.
            VmError: VM creation failed.
        """
        if not self._started:
            raise SandboxError("Scheduler not started. Use: async with Scheduler() as scheduler:")

        # Validate memory before acquiring resources
        memory = memory_mb or self.config.default_memory_mb
        if memory < constants.MIN_MEMORY_MB:
            raise VmConfigError(
                f"memory_mb={memory} is below minimum ({constants.MIN_MEMORY_MB}MB). "
                f"QEMU cannot boot a kernel with less than {constants.MIN_MEMORY_MB}MB of RAM."
            )

        if self._vm_manager is None or self._settings is None:
            raise SandboxError("Scheduler resources not initialized")

        # Ensure language is a Language enum (callers may pass raw strings)
        language = Language(language)
        packages = packages or []

        # Resolve port mappings (allocate ephemeral ports for those without explicit external)
        resolved_ports: list[ExposedPort] = []
        if expose_ports:
            resolved_ports = resolve_port_mappings(expose_ports)
            logger.info(
                "Resolved port mappings",
                extra={
                    "ports": [(p.internal, p.external) for p in resolved_ports],
                },
            )

        # Validate packages against allowlist
        if packages and self.config.enable_package_validation:
            await self._validate_packages(packages, language)

        # Check L2 snapshot cache for disk caching (only if packages specified)
        # Only use snapshots when packages need to be installed - empty packages case
        # benefits from overlay pool pre-caching which requires using the original base image
        snapshot_path: Path | None = None
        if packages:
            snapshot_path = await self._get_or_create_snapshot(language, packages, memory)
            logger.info(
                "Snapshot ready",
                extra={"snapshot_path": str(snapshot_path)},
            )

        # Try warm pool first (instant allocation)
        # Warm pool VMs don't have networking, so skip if networking is needed
        vm: QemuVM | None = None
        is_cold_boot = False
        needs_network: bool = allow_network or bool(resolved_ports)
        if self._warm_pool and not packages and not needs_network:
            vm = await self._warm_pool.get_vm(language, packages)

        # Cold boot if no warm VM available
        if vm is None:
            is_cold_boot = True

            # Auto-download base image if needed
            if self.config.auto_download_assets:
                from exec_sandbox.assets import fetch_base_image  # noqa: PLC0415

                await fetch_base_image(language)

            vm = await self._vm_manager.create_vm(
                language=language,
                tenant_id="exec-sandbox",
                task_id=task_id,
                backing_image=snapshot_path,
                memory_mb=memory,
                allow_network=allow_network,
                allowed_domains=allowed_domains,
                expose_ports=resolved_ports if resolved_ports else None,
            )

        return vm, resolved_ports, is_cold_boot

    def _create_settings(self) -> Settings:
        """Create Settings from SchedulerConfig.

        Bridges the public SchedulerConfig to internal Settings.
        Must be called after ensure_assets() has set self._images_dir.

        Returns:
            Settings instance configured from SchedulerConfig
        """
        if self._images_dir is None:
            raise RuntimeError("_create_settings called before ensure_assets")
        images_dir = self._images_dir

        # Kernels may be in images_dir directly or in a kernels subdirectory
        kernels_subdir = images_dir / "kernels"
        kernel_path = kernels_subdir if kernels_subdir.exists() else images_dir

        return Settings(
            base_images_dir=images_dir,
            kernel_path=kernel_path,
            snapshot_cache_dir=self.config.snapshot_cache_dir,
            s3_bucket=self.config.s3_bucket,
            s3_region=self.config.s3_region,
            max_concurrent_s3_uploads=self.config.max_concurrent_s3_uploads,
            memory_overcommit_ratio=self.config.memory_overcommit_ratio,
            cpu_overcommit_ratio=self.config.cpu_overcommit_ratio,
            host_memory_reserve_ratio=self.config.host_memory_reserve_ratio,
            resource_monitor_interval_seconds=self.config.resource_monitor_interval_seconds,
        )

    async def _validate_packages(self, packages: list[str], language: Language) -> None:
        """Validate packages against allowlist.

        Args:
            packages: List of package specifiers (e.g., ["pandas==2.2.0"])
            language: Programming language

        Raises:
            PackageNotAllowedError: Package not in allowlist
        """
        from exec_sandbox.package_validator import PackageValidator  # noqa: PLC0415

        validator = await PackageValidator.create()
        validator.validate(packages, language)

    async def _get_or_create_snapshot(self, language: Language, packages: list[str], memory_mb: int) -> Path:
        """Get cached snapshot or create new one with packages.

        Checks L2 (local qcow2) and L3 (S3) caches before building.

        Args:
            language: Programming language
            packages: List of packages to install
            memory_mb: VM memory in MB (included in cache key)

        Returns:
            Path to snapshot qcow2.

        Raises:
            SandboxError: Snapshot manager not configured or snapshot creation failed.
        """
        if not self._snapshot_manager:
            raise SandboxError("Packages requested but snapshot manager is not configured")

        snapshot_path = await self._snapshot_manager.get_or_create_snapshot(
            language=language,
            packages=packages,
            tenant_id="exec-sandbox",
            task_id=f"snapshot-{hash(tuple(sorted(packages)))}",
            memory_mb=memory_mb,
        )
        logger.debug(
            "Snapshot ready",
            extra={
                "language": language,
                "packages": packages,
                "path": str(snapshot_path),
            },
        )
        return snapshot_path

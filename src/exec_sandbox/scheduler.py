"""Scheduler - main entry point for exec-sandbox.

The Scheduler provides a simple API for executing code in isolated microVMs.
Inspired by aiojobs - explicit resource management via async context manager.

Architecture:
- VM lifecycle: VMs are NEVER reused. Each run() gets a fresh VM, destroyed after.
- Warm pool: Pre-started VMs waiting for commands (faster than cold boot).
- Snapshot cache: L0 (local memory snapshots) + L3 (S3) for package installation speedup.
- Backpressure: max_concurrent_vms prevents OOM from unbounded VM creation.

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
        max_concurrent_vms=20,
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

from exec_sandbox._logging import get_logger
from exec_sandbox.config import SchedulerConfig
from exec_sandbox.exceptions import PackageNotAllowedError, SandboxError
from exec_sandbox.models import ExecutionResult, Language, TimingBreakdown
from exec_sandbox.settings import Settings

if TYPE_CHECKING:
    from exec_sandbox.snapshot_manager import SnapshotManager
    from exec_sandbox.vm_manager import QemuVM, VmManager
    from exec_sandbox.warm_vm_pool import WarmVMPool

logger = get_logger(__name__)


class Scheduler:
    """Manages microVM lifecycle and code execution.

    The Scheduler is the main entry point for exec-sandbox. It handles:
    - VM pool management (max concurrent VMs, backpressure)
    - Warm VM pool (pre-booted VMs for instant execution)
    - Snapshot caching (L0 local + L3 S3)
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
        self._settings: Settings | None = None
        self._vm_manager: VmManager | None = None
        self._snapshot_manager: SnapshotManager | None = None
        self._warm_pool: WarmVMPool | None = None
        self._started = False
        self._semaphore: asyncio.Semaphore | None = None

    async def __aenter__(self) -> Self:
        """Start scheduler and initialize resources.

        Startup sequence:
        1. Download assets from GitHub Releases (if auto_download_assets=True)
        2. Create Settings from SchedulerConfig
        3. Initialize VmManager
        4. Initialize SnapshotManager (if S3 configured)
        5. Start WarmVMPool (if warm_pool_size > 0)

        Returns:
            Self for use in context

        Raises:
            SandboxError: Startup failed
            AssetDownloadError: Failed to download required assets
        """
        if self._started:
            raise SandboxError("Scheduler already started")

        logger.info(
            "Starting scheduler",
            extra={
                "max_concurrent_vms": self.config.max_concurrent_vms,
                "warm_pool_size": self.config.warm_pool_size,
                "s3_enabled": self.config.s3_bucket is not None,
                "auto_download_assets": self.config.auto_download_assets,
            },
        )

        # Download assets from GitHub Releases if needed
        if self.config.auto_download_assets:
            from exec_sandbox.assets import ensure_assets_available  # noqa: PLC0415

            # Pre-fetch kernel and gvproxy (base images are fetched on-demand)
            await ensure_assets_available()

        # Create Settings from SchedulerConfig
        self._settings = self._create_settings()

        # Initialize backpressure semaphore
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_vms)

        # Initialize VmManager
        from exec_sandbox.vm_manager import VmManager  # noqa: PLC0415

        self._vm_manager = VmManager(self._settings)

        # Initialize SnapshotManager (L0 local cache always available, L3 S3 optional)
        from exec_sandbox.snapshot_manager import SnapshotManager  # noqa: PLC0415

        self._snapshot_manager = SnapshotManager(self._settings, self._vm_manager)

        # Initialize WarmVMPool (optional)
        if self.config.warm_pool_size > 0:
            from exec_sandbox.warm_vm_pool import WarmVMPool  # noqa: PLC0415

            self._warm_pool = WarmVMPool(self._vm_manager, self.config)
            await self._warm_pool.startup()

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

        # Shutdown WarmVMPool first (drains VMs)
        if self._warm_pool:
            try:
                await self._warm_pool.shutdown()
            except (OSError, RuntimeError, TimeoutError) as e:
                logger.error("WarmVMPool shutdown error", extra={"error": str(e)})

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

        self._started = False
        logger.info("Scheduler shutdown complete")

    async def run(
        self,
        code: str,
        *,
        language: Language,
        packages: list[str] | None = None,
        timeout_seconds: int | None = None,
        memory_mb: int | None = None,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Execute code in an isolated microVM.

        Each call gets a fresh VM that is destroyed after execution.
        VMs are NEVER reused between runs (security guarantee).

        Args:
            code: Source code to execute.
            language: Programming language ("python" or "javascript").
            packages: Optional list of packages to install (e.g., ["pandas==2.2.0"]).
                Validated against allowlist if enable_package_validation=True.
            timeout_seconds: Execution timeout. Default: config.default_timeout_seconds.
            memory_mb: Guest VM memory in MB. Default: config.default_memory_mb.
            allow_network: Enable network access for the VM. Default: False.
            allowed_domains: Whitelist of domains if allow_network=True.
                If None/empty and allow_network=True, all domains allowed.
            env_vars: Environment variables to set in the VM.
            on_stdout: Callback for stdout chunks (streaming). Called as chunks arrive.
            on_stderr: Callback for stderr chunks (streaming). Called as chunks arrive.

        Returns:
            ExecutionResult with stdout, stderr, exit_code, and timing info.

        Raises:
            SandboxError: Scheduler not started.
            PackageNotAllowedError: Package not in allowlist.
            VmError: VM creation or execution failed.
            VmTimeoutError: Execution exceeded timeout.

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

            # With streaming
            result = await scheduler.run(
                "for i in range(5): print(i)",
                language="python",
                on_stdout=lambda chunk: print(f"[OUT] {chunk}"),
            )
            ```
        """
        if not self._started:
            raise SandboxError("Scheduler not started. Use: async with Scheduler() as scheduler:")

        # Type narrowing (guaranteed by _started check)
        if self._vm_manager is None or self._semaphore is None or self._settings is None:
            raise SandboxError("Scheduler resources not initialized")

        # Apply defaults
        timeout = timeout_seconds or self.config.default_timeout_seconds
        memory = memory_mb or self.config.default_memory_mb
        packages = packages or []

        # Validate packages against allowlist
        if packages and self.config.enable_package_validation:
            self._validate_packages(packages, language)

        # Determine snapshot path (if packages specified)
        snapshot_path: Path | None = None
        has_memory_snapshot = False
        if packages and self._snapshot_manager:
            # Check snapshot cache for pre-built image with packages
            snapshot_path, has_memory_snapshot = await self._get_or_create_snapshot(language, packages)
            # TODO: Pass has_memory_snapshot to create_vm for fast restore via -loadvm
            _ = has_memory_snapshot  # Suppress unused warning until fully integrated

        # Acquire semaphore (backpressure)
        async with self._semaphore:
            vm: QemuVM | None = None
            run_start_time = asyncio.get_event_loop().time()
            is_cold_boot = False
            try:
                # Try warm pool first (instant allocation)
                if self._warm_pool and not packages:
                    lang_enum = Language(language)
                    vm = await self._warm_pool.get_vm(lang_enum, packages)

                # Cold boot if no warm VM available
                if vm is None:
                    is_cold_boot = True
                    vm = await self._vm_manager.create_vm(
                        language=language,
                        tenant_id="exec-sandbox",
                        task_id=f"run-{id(code)}",
                        snapshot_path=snapshot_path,
                        memory_mb=memory,
                        allow_network=allow_network,
                        allowed_domains=allowed_domains,
                    )

                # Execute code
                execute_start_time = asyncio.get_event_loop().time()
                result = await vm.execute(
                    code=code,
                    language=language,
                    timeout_seconds=timeout,
                    env_vars=env_vars,
                    on_stdout=on_stdout,
                    on_stderr=on_stderr,
                )
                execute_end_time = asyncio.get_event_loop().time()

                # Attach timing breakdown for cold boots
                if is_cold_boot and vm.setup_ms is not None and vm.boot_ms is not None:
                    execute_ms = int((execute_end_time - execute_start_time) * 1000)
                    total_ms = int((execute_end_time - run_start_time) * 1000)
                    result = ExecutionResult(
                        stdout=result.stdout,
                        stderr=result.stderr,
                        exit_code=result.exit_code,
                        execution_time_ms=result.execution_time_ms,
                        external_cpu_time_ms=result.external_cpu_time_ms,
                        external_memory_peak_mb=result.external_memory_peak_mb,
                        timing=TimingBreakdown(
                            setup_ms=vm.setup_ms,
                            boot_ms=vm.boot_ms,
                            execute_ms=execute_ms,
                            total_ms=total_ms,
                        ),
                    )

                return result

            finally:
                # Always destroy VM (never reused)
                if vm is not None:
                    await self._vm_manager.destroy_vm(vm)

    def _create_settings(self) -> Settings:
        """Create Settings from SchedulerConfig.

        Bridges the public SchedulerConfig to internal Settings.

        Returns:
            Settings instance configured from SchedulerConfig
        """
        # Get images directory (auto-detect if not configured)
        # When auto_download_assets=True, assets were already downloaded in __aenter__
        # so we don't check existence again (they should exist now)
        images_dir = self.config.get_images_dir(check_exists=not self.config.auto_download_assets)

        # Kernels may be in images_dir directly or in a kernels subdirectory
        kernels_subdir = images_dir / "kernels"
        kernel_path = kernels_subdir if kernels_subdir.exists() else images_dir

        return Settings(
            base_images_dir=images_dir,
            kernel_path=kernel_path,
            max_concurrent_vms=self.config.max_concurrent_vms,
            snapshot_cache_dir=self.config.snapshot_cache_dir,
            s3_bucket=self.config.s3_bucket,
            s3_region=self.config.s3_region,
            max_concurrent_s3_uploads=self.config.max_concurrent_s3_uploads,
        )

    def _validate_packages(self, packages: list[str], language: Language) -> None:
        """Validate packages against allowlist.

        Args:
            packages: List of package specifiers (e.g., ["pandas==2.2.0"])
            language: Programming language

        Raises:
            PackageNotAllowedError: Package not in allowlist
        """
        from exec_sandbox.package_validator import PackageValidator  # noqa: PLC0415

        validator = PackageValidator()
        for package in packages:
            # Extract package name (strip version specifier)
            name = package.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("[")[0]
            if not validator.is_allowed(name, language.value):
                raise PackageNotAllowedError(
                    f"Package '{name}' not in allowlist for {language.value}",
                    context={"package": name, "language": language.value},
                )

    async def _get_or_create_snapshot(self, language: str, packages: list[str]) -> tuple[Path | None, bool]:
        """Get cached snapshot or create new one with packages.

        Checks L0 (memory snapshot) and L3 (S3) caches before building.

        Args:
            language: Programming language
            packages: List of packages to install

        Returns:
            Tuple of (snapshot_path, has_memory_snapshot):
            - (path, True) if snapshot has memory state (use -loadvm for fast restore)
            - (path, False) if snapshot is disk-only (cold boot required)
            - (None, False) on error (graceful degradation)
        """
        if not self._snapshot_manager:
            return (None, False)

        try:
            snapshot_path, has_memory_snapshot = await self._snapshot_manager.get_or_create_snapshot(
                language=Language(language),
                packages=packages,
                tenant_id="exec-sandbox",
                task_id=f"snapshot-{hash(tuple(sorted(packages)))}",
            )
            logger.debug(
                "Snapshot ready",
                extra={
                    "language": language,
                    "packages": packages,
                    "path": str(snapshot_path),
                    "has_memory_snapshot": has_memory_snapshot,
                },
            )
            return (snapshot_path, has_memory_snapshot)
        except (OSError, RuntimeError, TimeoutError, ConnectionError) as e:
            # Graceful degradation: log error, continue without snapshot
            logger.warning(
                "Snapshot creation failed, continuing without cache",
                extra={"language": language, "packages": packages, "error": str(e)},
            )
            return (None, False)

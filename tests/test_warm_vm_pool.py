"""Tests for WarmVMPool.

Unit tests: Pool data structures, config handling.
Integration tests: Real VM pool operations (requires QEMU + images).
"""

from pathlib import Path

from exec_sandbox import constants
from exec_sandbox.models import Language

# ============================================================================
# Unit Tests - No QEMU needed
# ============================================================================


class TestWarmVMPoolConfig:
    """Tests for WarmVMPool configuration."""

    def test_pool_size_calculation(self) -> None:
        """Pool size is 25% of max_concurrent_vms."""
        # The calculation: max(1, int(max_concurrent_vms * 0.25))

        # max_concurrent_vms=10 → pool_size=2
        expected = max(1, int(10 * constants.WARM_POOL_SIZE_RATIO))
        assert expected == 2

        # max_concurrent_vms=100 → pool_size=25
        expected = max(1, int(100 * constants.WARM_POOL_SIZE_RATIO))
        assert expected == 25

        # max_concurrent_vms=1 → pool_size=1 (minimum)
        expected = max(1, int(1 * constants.WARM_POOL_SIZE_RATIO))
        assert expected == 1

    def test_warm_pool_languages(self) -> None:
        """Warm pool supports python and javascript."""
        assert Language.PYTHON in constants.WARM_POOL_LANGUAGES
        assert Language.JAVASCRIPT in constants.WARM_POOL_LANGUAGES
        assert len(constants.WARM_POOL_LANGUAGES) == 2


class TestLanguageEnum:
    """Tests for Language enum."""

    def test_language_values(self) -> None:
        """Language enum has expected values."""
        assert Language.PYTHON.value == "python"
        assert Language.JAVASCRIPT.value == "javascript"

    def test_language_from_string(self) -> None:
        """Language can be created from string."""
        assert Language("python") == Language.PYTHON
        assert Language("javascript") == Language.JAVASCRIPT


# ============================================================================
# Integration Tests - Require QEMU + Images
# ============================================================================


class TestWarmVMPoolIntegration:
    """Integration tests for WarmVMPool with real QEMU VMs."""

    async def test_pool_startup_shutdown(self, images_dir: Path) -> None:
        """Pool starts and shuts down cleanly."""
        from exec_sandbox.settings import Settings
        from exec_sandbox.vm_manager import VmManager
        from exec_sandbox.warm_vm_pool import WarmVMPool

        settings = Settings(
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            max_concurrent_vms=4,
        )
        vm_manager = VmManager(settings)

        config = type("Config", (), {"max_concurrent_vms": 4})()
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

        # Pools should be populated
        assert pool.pools[Language.PYTHON].qsize() > 0

        await pool.shutdown()

        # Pools should be empty
        assert pool.pools[Language.PYTHON].qsize() == 0
        assert pool.pools[Language.JAVASCRIPT].qsize() == 0

    async def test_get_vm_from_pool(self, images_dir: Path) -> None:
        """Get VM from warm pool."""
        from exec_sandbox.settings import Settings
        from exec_sandbox.vm_manager import VmManager
        from exec_sandbox.warm_vm_pool import WarmVMPool

        settings = Settings(
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            max_concurrent_vms=4,
        )
        vm_manager = VmManager(settings)

        config = type("Config", (), {"max_concurrent_vms": 4})()
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

        try:
            # Get VM from pool (should be instant)
            vm = await pool.get_vm(Language.PYTHON, packages=[])

            assert vm is not None
            assert vm.vm_id is not None

            # Destroy VM after use
            await vm_manager.destroy_vm(vm)

        finally:
            await pool.shutdown()

    async def test_get_vm_with_packages_returns_none(self, images_dir: Path) -> None:
        """Get VM with packages returns None (not eligible for warm pool)."""
        from exec_sandbox.settings import Settings
        from exec_sandbox.vm_manager import VmManager
        from exec_sandbox.warm_vm_pool import WarmVMPool

        settings = Settings(
            base_images_dir=images_dir,
            kernel_path=images_dir / "kernels" if (images_dir / "kernels").exists() else images_dir,
            max_concurrent_vms=4,
        )
        vm_manager = VmManager(settings)

        config = type("Config", (), {"max_concurrent_vms": 4})()
        pool = WarmVMPool(vm_manager, config)

        await pool.startup()

        try:
            # Get VM with packages - should return None
            vm = await pool.get_vm(Language.PYTHON, packages=["pandas==2.0.0"])
            assert vm is None

        finally:
            await pool.shutdown()

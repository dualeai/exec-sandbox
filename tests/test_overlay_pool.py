"""Tests for OverlayPool.

Test philosophy:
- Unit tests: Pure logic only (no I/O, no mocks needed)
- Error tests: Mock only to simulate failures that can't be triggered otherwise
- Integration tests: Real qemu-img, real files, real code paths
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from .conftest import skip_unless_hwaccel

# ============================================================================
# Unit Tests - Pure Logic (no I/O, no mocks)
# ============================================================================


class TestOverlayPoolPureLogic:
    """Tests for pure logic - no I/O needed."""

    def test_pool_size_zero_disables_pool(self, tmp_path: Path) -> None:
        """Pool size 0 means pool is disabled."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool = OverlayPool(pool_size=0, pool_dir=tmp_path / "pool")
        assert pool.pool_size == 0

    def test_negative_pool_size_treated_as_disabled(self, tmp_path: Path) -> None:
        """Negative pool size behaves like pool_size=0."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool = OverlayPool(pool_size=-5, pool_dir=tmp_path / "pool")
        assert pool.pool_size == -5  # Stored as-is, but treated as disabled

    async def test_acquire_before_start_fails(self, tmp_path: Path) -> None:
        """Acquire before start raises RuntimeError - daemon required."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")
        # No start() called - daemon not started
        target = tmp_path / "target.qcow2"

        with pytest.raises(RuntimeError, match="Daemon must be started"):
            await pool.acquire(Path("/fake/base.qcow2"), target)

    async def test_start_with_zero_pool_size_starts_daemon_only(self, tmp_path: Path) -> None:
        """start() with pool_size=0 starts daemon but doesn't create pool directory or tasks."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool_dir = tmp_path / "overlay-pool"
        pool = OverlayPool(pool_size=0, pool_dir=pool_dir)

        await pool.start([Path("/fake/base.qcow2")])

        # Daemon is started (for on-demand creation in acquire)
        assert pool._started
        assert pool._daemon is not None
        # But no pool directory or replenish tasks
        assert not pool_dir.exists()
        assert len(pool._replenish_tasks) == 0
        await pool.stop()

    async def test_start_with_negative_pool_size_starts_daemon_only(self, tmp_path: Path) -> None:
        """start() with negative pool_size starts daemon but no pre-creation."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=-5, pool_dir=pool_dir)

        await pool.start([Path("/fake/base.qcow2")])

        assert pool._started
        assert pool._daemon is not None
        assert not pool_dir.exists()
        await pool.stop()

    async def test_empty_base_images_list(self, tmp_path: Path) -> None:
        """start() with empty base images list creates directory but no pools."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")
        await pool.start([])  # Empty list

        assert len(pool._pools) == 0
        assert pool._started
        await pool.stop()

    async def test_double_stop_is_safe(self, tmp_path: Path) -> None:
        """Calling stop() twice doesn't error (idempotent)."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool = OverlayPool(pool_size=0, pool_dir=tmp_path / "pool")
        await pool.start([])

        await pool.stop()
        await pool.stop()  # Should not raise

    async def test_double_start_raises_error(self, tmp_path: Path) -> None:
        """Calling start() twice without stop() raises RuntimeError."""
        import pytest

        from exec_sandbox.overlay_pool import OverlayPool

        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")
        await pool.start([])

        with pytest.raises(RuntimeError, match="already started"):
            await pool.start([])

        await pool.stop()

    async def test_start_restart_after_stop(self, tmp_path: Path) -> None:
        """Pool can restart after stop (shutdown_event is cleared)."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")

        # First lifecycle
        await pool.start([])
        assert pool._started
        await pool.stop()
        assert not pool._started

        # Second lifecycle - should work
        await pool.start([])
        assert pool._started
        await pool.stop()

    async def test_acquire_existing_target_raises_error(self, tmp_path: Path) -> None:
        """Acquire raises FileExistsError if target_path already exists."""
        import pytest

        from exec_sandbox.overlay_pool import OverlayPool

        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")
        target = tmp_path / "existing.qcow2"
        target.write_text("existing content")

        with pytest.raises(FileExistsError, match="already exists"):
            await pool.acquire(Path("/fake/base.qcow2"), target)

    async def test_mkdir_permission_error_disables_pool_precreation(self, tmp_path: Path) -> None:
        """Permission error during mkdir disables pool pre-creation but daemon stays started."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")

        with patch("aiofiles.os.makedirs", side_effect=PermissionError("Access denied")):
            await pool.start([Path("/fake/base.qcow2")])

        # Daemon is started (for on-demand creation)
        assert pool._started
        assert pool._daemon is not None
        # But no pools are created due to mkdir failure
        assert len(pool._pools) == 0
        await pool.stop()


# ============================================================================
# Error Handling Tests - Mocks needed to simulate failures
# ============================================================================


class TestOverlayPoolErrorHandling:
    """Tests for error handling - mocks needed to simulate failures."""

    async def test_rename_failure_falls_back_to_ondemand(self, tmp_path: Path) -> None:
        """Failed rename (cross-filesystem) cleans up and creates on-demand.

        Note: Uses internal state setup because triggering real rename failure
        requires cross-filesystem setup which is environment-dependent.
        """
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.qemu_storage_daemon import QemuStorageDaemon

        pool_dir = tmp_path / "pool"
        pool_dir.mkdir(parents=True)
        pool = OverlayPool(pool_size=1, pool_dir=pool_dir)

        # Setup: create a file in the pool queue and mock daemon
        base_image = Path("/fake/base.qcow2")
        pool._pools[str(base_image)] = asyncio.Queue(maxsize=1)
        overlay = pool_dir / "test.qcow2"
        overlay.write_text("content")
        await pool._pools[str(base_image)].put(overlay)
        pool._started = True

        # Mock daemon for on-demand creation
        mock_daemon = AsyncMock(spec=QemuStorageDaemon)
        pool._daemon = mock_daemon

        # Test: rename fails, should cleanup and fall back to on-demand via daemon
        with patch("aiofiles.os.rename", side_effect=OSError("Cross-device link")):
            result = await pool.acquire(base_image, tmp_path / "target.qcow2")

        assert result is False  # Not from pool (created on-demand)
        assert not overlay.exists()  # Orphaned overlay cleaned up
        mock_daemon.create_overlay.assert_called_once()  # Fell back to daemon
        await pool.stop()

    async def test_stop_handles_rmtree_failure(self, tmp_path: Path) -> None:
        """stop() completes even if directory cleanup fails."""
        from exec_sandbox.overlay_pool import OverlayPool

        pool_dir = tmp_path / "pool"
        pool = OverlayPool(pool_size=5, pool_dir=pool_dir)

        # Use empty startup (no base images = no qemu-img calls needed)
        await pool.start([])

        # Manually create directory to simulate state after real startup
        pool_dir.mkdir(parents=True, exist_ok=True)

        with patch("shutil.rmtree", side_effect=OSError("Permission denied")):
            await pool.stop()  # Should not raise

        assert not pool._started


# ============================================================================
# Integration Tests - Real qemu-img, real files, real code
# ============================================================================


@skip_unless_hwaccel
class TestOverlayPoolIntegration:
    """Integration tests with real qemu-img - no mocking."""

    async def test_full_lifecycle(self, vm_settings, tmp_path: Path) -> None:
        """Test complete lifecycle: start → acquire → stop."""
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        # Verify pool has overlays
        key = str(base_image.resolve())
        assert pool._pools[key].qsize() == 2

        # Acquire one
        target = tmp_path / "acquired.qcow2"
        result = await pool.acquire(base_image, target)

        assert result is True
        assert target.exists()
        assert pool._pools[key].qsize() == 1  # One less in pool

        await pool.stop()
        assert not (tmp_path / "pool").exists()  # Cleaned up

    async def test_acquire_with_zero_pool_size_creates_on_demand(self, vm_settings, tmp_path: Path) -> None:
        """Test acquire works with pool_size=0 (CLI single-VM mode).

        This is a regression test for the bug where pool_size=0, and start()
        didn't initialize the daemon, causing acquire() to fail with
        "Daemon must be started before acquire".
        """
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        # pool_size=0
        pool = OverlayPool(pool_size=0, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        # Daemon should be started even with pool_size=0
        assert pool._started
        assert pool._daemon is not None
        # No pre-created overlays (pool_size=0)
        assert len(pool._pools) == 0

        # Acquire should work via on-demand creation
        target = tmp_path / "acquired.qcow2"
        result = await pool.acquire(base_image, target)

        # Returns False because created on-demand (not from pool)
        assert result is False
        assert target.exists()

        await pool.stop()

    async def test_acquired_overlay_has_correct_backing_file(self, vm_settings, tmp_path: Path) -> None:
        """Acquired overlay references correct base image."""
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        pool = OverlayPool(pool_size=1, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        target = tmp_path / "acquired.qcow2"
        await pool.acquire(base_image, target)

        # Verify backing file using qemu-img info
        proc = await asyncio.create_subprocess_exec(
            "qemu-img",
            "info",
            str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        assert str(base_image) in stdout.decode()

        await pool.stop()

    async def test_pool_exhaustion_creates_ondemand(self, vm_settings, tmp_path: Path) -> None:
        """Acquiring more than pool_size creates on-demand (returns False but succeeds)."""
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        # Acquire all 2 from pool, then 1 on-demand
        t1 = tmp_path / "t1.qcow2"
        t2 = tmp_path / "t2.qcow2"
        t3 = tmp_path / "t3.qcow2"

        assert await pool.acquire(base_image, t1) is True  # From pool
        assert await pool.acquire(base_image, t2) is True  # From pool
        assert await pool.acquire(base_image, t3) is False  # Created on-demand

        # All 3 overlays should exist and be valid
        assert t1.exists()
        assert t2.exists()
        assert t3.exists()

        await pool.stop()

    async def test_concurrent_acquires_all_succeed(self, vm_settings, tmp_path: Path) -> None:
        """Concurrent acquires all succeed (some from pool, rest on-demand)."""
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        pool = OverlayPool(pool_size=5, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        # 10 concurrent acquires for 5 in pool
        targets = [tmp_path / f"target-{i}.qcow2" for i in range(10)]
        results = await asyncio.gather(*[pool.acquire(base_image, t) for t in targets])

        # 5 from pool (True), 5 on-demand (False)
        assert sum(results) == 5

        # All 10 overlays should exist and be valid
        assert all(t.exists() for t in targets)
        sizes = [t.stat().st_size for t in targets]
        assert all(s > 0 for s in sizes)

        await pool.stop()

    async def test_multiple_base_images(self, vm_settings, tmp_path: Path) -> None:
        """Pool handles multiple different base images."""
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        python_base = vm_manager.get_base_image("python")
        js_base = vm_manager.get_base_image("javascript")

        pool = OverlayPool(pool_size=2, pool_dir=tmp_path / "pool")
        await pool.start([python_base, js_base])

        # Should have pools for both
        assert pool._pools[str(python_base.resolve())].qsize() == 2
        assert pool._pools[str(js_base.resolve())].qsize() == 2

        # Acquire from each
        py_target = tmp_path / "py.qcow2"
        js_target = tmp_path / "js.qcow2"

        assert await pool.acquire(python_base, py_target) is True
        assert await pool.acquire(js_base, js_target) is True

        # Verify each has correct backing file
        for target, base in [(py_target, python_base), (js_target, js_base)]:
            proc = await asyncio.create_subprocess_exec(
                "qemu-img",
                "info",
                str(target),
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            assert str(base) in stdout.decode()

        await pool.stop()

    async def test_vm_boots_with_pooled_overlay(self, vm_manager, vm_settings) -> None:
        """Full integration: VM boots successfully with pooled overlay."""
        from exec_sandbox.models import Language

        # vm_manager fixture already calls start() and stop()
        vm = await vm_manager.create_vm(
            language=Language.PYTHON,
            tenant_id="test",
            task_id="pool-test",
        )

        try:
            result = await vm.execute("print('hello from pool')", timeout_seconds=30)
            assert "hello from pool" in result.stdout
        finally:
            await vm_manager.destroy_vm(vm)

    async def test_fallback_to_ondemand_when_pool_exhausted(self, make_vm_settings, tmp_path: Path) -> None:
        """VM still boots when pool is empty (fallback to _create_overlay)."""
        from exec_sandbox.models import Language
        from exec_sandbox.vm_manager import VmManager

        # pool_size=2
        settings = make_vm_settings()

        async with VmManager(settings) as vm_manager:
            # Create 3 VMs (exhausts pool of 2, forces 1 on-demand)
            vms = []
            for i in range(3):
                vm = await vm_manager.create_vm(
                    language=Language.PYTHON,
                    tenant_id="test",
                    task_id=f"exhaust-{i}",
                )
                vms.append(vm)

            assert len(vms) == 3

            # All VMs should work
            for vm in vms:
                result = await vm.execute("print(1)", timeout_seconds=30)
                assert "1" in result.stdout

            for vm in vms:
                await vm_manager.destroy_vm(vm)

    async def test_get_stats(self, vm_settings, tmp_path: Path) -> None:
        """get_stats returns accurate pool sizes."""
        from exec_sandbox.overlay_pool import OverlayPool
        from exec_sandbox.vm_manager import VmManager

        vm_manager = VmManager(vm_settings)
        base_image = vm_manager.get_base_image("python")

        pool = OverlayPool(pool_size=3, pool_dir=tmp_path / "pool")
        await pool.start([base_image])

        stats = pool.get_stats()
        assert stats[str(base_image.resolve())] == 3

        # Acquire one
        await pool.acquire(base_image, tmp_path / "t.qcow2")

        stats = pool.get_stats()
        assert stats[str(base_image.resolve())] == 2

        await pool.stop()

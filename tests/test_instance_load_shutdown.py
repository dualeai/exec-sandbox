"""A load's stream files stay open while its apply thread writes, also when the event loop
shuts down (asyncio.run cancels every task left: Ctrl-C on `sbx run --instance`).

A file descriptor closed under a running writer is a number the process can reuse at once:
another thread's new file would then get the guest's stream bytes.
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
from typing import TYPE_CHECKING

from exec_sandbox.instance import manager as instance_manager
from exec_sandbox.instance.keyring import Keyring
from exec_sandbox.instance.manager import InstanceManager
from exec_sandbox.instance.s3_store import S3InstanceStore
from tests.instance_support import BLOCK, NO_VM_TIMEOUT, PYTHON, Saver, SparseImage, no_packages, random_block

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from tests.instance_support import InstanceEnv

pytestmark = NO_VM_TIMEOUT


async def test_loop_shutdown_during_apply_keeps_the_stream_files_open(
    instance_env: InstanceEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real InstanceManager.load in an asyncio.run of its own (another thread: the store of
    the instance_env serves it from the test's loop) is left running when that loop ends.
    While asyncio.run cancels it, the apply thread still writes: each of its stream files is
    still the file it opened."""
    env = instance_env
    rng = random.Random("load-shutdown")
    ram = SparseImage(size=4 * BLOCK)
    for block_no in range(4):
        ram.put(block_no, random_block(rng))
    vm = await env.new_vm(ram, SparseImage(size=BLOCK), spec=PYTHON)
    instance_id = (await Saver(env, vm, manager=env.manager, state=None).save()).instance_id

    applying = threading.Event()
    cancelled = threading.Event()  # set once asyncio.run has cancelled the load task (its finally ran)
    seen: list[tuple[int, int]] = []  # (inode at the start of the apply, inode after the loop stopped)
    real_apply = instance_manager.apply_object

    def slow_apply(src: Path, *, targets: dict[int, int], **kwargs: object) -> None:
        start = {fd: os.fstat(fd).st_ino for fd in targets.values()}
        applying.set()
        assert cancelled.wait(5)  # an event, never a delay: the verdict must not depend on host load
        for fd, inode in start.items():
            try:
                seen.append((inode, os.fstat(fd).st_ino))
            except OSError:
                seen.append((inode, -1))  # closed under the writer
        real_apply(src, targets=targets, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(instance_manager, "apply_object", slow_apply)
    work_dir, asset_dir = tmp_path / "other-worker" / "work", tmp_path / "other-worker" / "assets"
    work_dir.mkdir(mode=0o700, parents=True)
    asset_dir.mkdir(mode=0o700)

    async def app() -> None:
        async with S3InstanceStore.open(
            bucket=env.proxy.bucket.name,
            prefix=env.proxy.prefix,
            endpoint_url=env.proxy.endpoint_url,
            region="us-east-1",
        ) as store:
            manager = InstanceManager(
                store=store,
                keyring=Keyring.parse(env.keyring),
                work_dir=work_dir,
                asset_dir=asset_dir,
                vm_manager=env.vm_manager,  # type: ignore[arg-type]
            )
            task = asyncio.ensure_future(manager.load(instance_id, packages_image=no_packages))
            task.add_done_callback(lambda _: cancelled.set())  # left running on purpose: asyncio.run cancels it
            await asyncio.to_thread(applying.wait)

    await asyncio.to_thread(asyncio.run, app())

    assert len(seen) == 2  # the RAM and home stream files
    assert all(before == after for before, after in seen), seen

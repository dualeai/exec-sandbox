"""A load's stream files stay open while its apply thread writes, also when the event loop
shuts down (asyncio.run cancels every task left: Ctrl-C on `sbx run --instance`).

A file descriptor closed under a running writer is a number the process can reuse at once:
another thread's new file would then get the guest's stream bytes.
"""

import asyncio
import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from exec_sandbox.instance import ids
from exec_sandbox.instance import manager as instance_manager
from exec_sandbox.instance.manager import InstanceManager
from exec_sandbox.instance.models import STREAM_HOME, STREAM_VMSTATE, ChainEntry
from exec_sandbox.instance.pack import write_object

_BLOCK = 64 * 1024
_KEY = os.urandom(32)


def test_loop_shutdown_during_apply_keeps_the_stream_files_open(tmp_path: Path) -> None:
    instance_id, commit_id = ids.new_id(), ids.new_id()
    ram = tmp_path / "ram"
    ram.write_bytes(os.urandom(4 * _BLOCK))
    home = tmp_path / "home"
    home.write_bytes(b"")
    obj = tmp_path / "obj.f"
    size = write_object(
        obj,
        data_key=_KEY,
        instance_id=instance_id,
        commit_id=commit_id,
        streams={STREAM_VMSTATE: (ram, None), STREAM_HOME: (home, None)},
    )

    async def download(_key: str, *, dest: Path, max_bytes: int, expected_size: int) -> bool:
        shutil.copyfile(obj, dest)
        return True

    applying = threading.Event()
    cancelled = threading.Event()  # set once asyncio.run has cancelled the load task (its finally ran)
    seen: list[tuple[int, int]] = []  # (inode at the start of the apply, inode after the loop stopped)
    real_apply = instance_manager.apply_object

    def slow_apply(src: Path, *, targets: dict[int, int], **kwargs: object) -> None:
        start = {fd: os.fstat(fd).st_ino for fd in targets.values()}
        applying.set()
        assert cancelled.wait(10)  # an event, never a delay: the verdict must not depend on host load
        for fd, inode in start.items():
            try:
                seen.append((inode, os.fstat(fd).st_ino))
            except OSError:
                seen.append((inode, -1))  # closed under the writer
        real_apply(src, targets=targets, **kwargs)  # type: ignore[arg-type]

    folder = tmp_path / "work"
    folder.mkdir()
    # The private _download is called directly: no public path reaches the apply step
    # without a sealed head, a keyring, the QEMU floor and the host checks. It reads
    # only the store's download, the header's two ids and the body's chain.
    store = SimpleNamespace(download=download)
    header = SimpleNamespace(instance_id=instance_id, commit_id=commit_id)
    entry = ChainEntry(commit_id=commit_id, kind="full", size=size, stream_sizes=(4 * _BLOCK, 0), etag="e")
    body = SimpleNamespace(chain=(entry,))

    async def app() -> None:
        load = InstanceManager._download(SimpleNamespace(store=store), header, body, _KEY, folder)  # type: ignore[arg-type]
        task = asyncio.ensure_future(load)  # left running on purpose: asyncio.run cancels it
        task.add_done_callback(lambda _: cancelled.set())
        await asyncio.to_thread(applying.wait)

    with patch.object(instance_manager, "apply_object", slow_apply):
        asyncio.run(app())

    assert len(seen) == 2
    assert all(before == after for before, after in seen), seen

"""Files of an L1 memory snapshot entry: their names next to the vmstate, the JSON
metadata sidecar, and the RAM folders in memory (/dev/shm) of template save VMs.

VmManager (restore_vm, the sweep of start) and MemorySnapshotManager (the L1 cache)
both use it; it imports neither.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

from exec_sandbox._logging import get_logger
from exec_sandbox.exceptions import ConfigError
from exec_sandbox.private_run_dir import PrivateRunDir, new_folder_name, remove_dead_folders

logger = get_logger(__name__)

_LIVE_RAM_ROOT: Final = Path("/dev/shm")  # noqa: S108 - only a folder of this user under it (PrivateRunDir checks it)
"""The memory file system (tmpfs) of Linux where a template save VM keeps its RAM file."""


def meta_path_from_vmstate(vmstate_path: Path) -> Path:
    """JSON metadata sidecar (SnapshotMetadata) of an L1 entry: sibling of the vmstate
    with ``.vmstate.meta`` extension."""
    return vmstate_path.with_suffix(".vmstate.meta")


def ram_path_from_vmstate(vmstate_path: Path) -> Path:
    """Derive RAM template file path from vmstate path.

    The RAM file is a sibling of the vmstate with ``.ram`` extension.
    Used by vm_manager to pass mem_path to build_qemu_cmd for COW restore.
    """
    # vmstate_path = /cache/l1-python-v1.2-abcd1234.vmstate
    # ram_path     = /cache/l1-python-v1.2-abcd1234.ram
    return vmstate_path.with_suffix(".ram")


def home_path_from_vmstate(vmstate_path: Path) -> Path:
    """Home disk of an L1 entry: sibling of the vmstate with ``.home`` extension.

    A restored VM resumes with a page cache that refers to these blocks, so
    it gets a clone of this disk, never a blank one. VmManager.restore_vm
    also uses it to find a saved instance's home disk: InstanceManager names
    its files the same way.
    """
    return vmstate_path.with_suffix(".home")


@dataclass(frozen=True)
class SnapshotMetadata:
    """The JSON sidecar of an L1 vmstate: what a check of the entry and a restore read. The
    cache key holds what the entry was made from (compute_cache_key)."""

    vmstate_sha256: str
    """_vmstate_digest (memory_snapshot_manager) of the vmstate; empty: not checked."""
    use_template: bool
    """The guest RAM is the {key}.ram template (memory-backend-file), not in the vmstate."""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> SnapshotMetadata | None:
        """Parse JSON text into SnapshotMetadata, or None when it is not a JSON object."""
        try:
            parsed = json.loads(text)
        except ValueError:  # json.JSONDecodeError too
            return None
        if not isinstance(parsed, dict):
            return None
        fields = cast("dict[str, object]", parsed)
        return cls(
            vmstate_sha256=str(fields.get("vmstate_sha256", "")),
            use_template=bool(fields.get("use_template", False)),
        )


def _live_ram_root() -> Path:
    """The folder of this user in /dev/shm: the RAM folders of its template save VMs are in it."""
    return _LIVE_RAM_ROOT / f"exec-sandbox-{os.geteuid()}"


def remove_dead_live_ram_dirs() -> None:
    """Remove the RAM folders of template save VMs whose process died (a crash, SIGKILL):
    each holds up to a whole guest RAM in memory, until the next template save of this user
    would remove it. VmManager.start runs it, after it killed the VMs of dead processes.
    Sync (a listing): async callers use asyncio.to_thread. With no /dev/shm folder
    (macOS) there is nothing to remove. A folder that is not private is logged, not raised.

    Raises:
        OSError: the folder of this user in /dev/shm cannot be opened or listed (remove_dead_folders)
    """
    try:
        remove_dead_folders(_live_ram_root())  # checks the root before it removes anything in it
    except ConfigError as e:
        logger.warning("RAM folders of stopped L1 saves stay in /dev/shm", extra={"error": str(e)})


def open_live_ram_dir(*, size: int) -> PrivateRunDir | None:
    """A new folder in memory for the RAM file of a template save VM (`size` bytes), or
    None: the VM then maps its RAM file in the cache folder.

    On a disk, each page the guest writes is dirty page cache that writeback writes again
    and again, and the kernel stalls the vCPUs meanwhile (balance_dirty_pages, waits for
    writeback). Measured on EBS: a Python save VM wrote 131 MB/s for 10-15 s, and its save
    took 12-17 s, not 1 s. In memory (tmpfs) its RAM is shared memory, charged to the VM's
    memory cgroup as anonymous RAM is; _publish_ram writes it to the cache once.

    /dev/shm is the folder of POSIX shared memory: a memory filesystem wherever the host
    keeps the convention, whatever its type (what the host mounts there is its choice; a
    disk there is as slow as the cache folder, never wrong). None when there is no such
    folder (macOS: a POSIX shared memory object has no path QEMU can open; there the save
    takes 0.6 s with the file on the disk), when /dev/shm/exec-sandbox-<uid> is not a private folder of this
    user (anyone can make it first), or when /dev/shm has less than `size` bytes free (a
    container's /dev/shm is 64 MB by default). The folders of dead processes go first:
    their RAM files would hold memory until a reboot. None too when /dev/shm cannot be used
    (an OSError: no access, a full or read-only filesystem): the save is slower, never lost.
    Sync: metadata calls.
    """
    if not _LIVE_RAM_ROOT.is_dir():
        return None
    root = _live_ram_root()
    try:
        live_dir = PrivateRunDir.open(root, name=new_folder_name())  # checks the root before the sweep
    except (ConfigError, OSError) as e:
        logger.warning("L1 template RAM goes to the cache folder, the save VM runs slower", extra={"error": str(e)})
        return None
    try:
        remove_dead_folders(root)
        st = os.statvfs(_LIVE_RAM_ROOT)
    except OSError as e:
        live_dir.close()
        logger.warning("L1 template RAM goes to the cache folder, the save VM runs slower", extra={"error": str(e)})
        return None
    except BaseException:
        live_dir.close()
        raise
    if st.f_bavail * st.f_frsize < size:
        live_dir.close()
        logger.warning(
            "L1 template RAM goes to the cache folder: /dev/shm is too small, the save VM runs slower",
            extra={"free_mb": st.f_bavail * st.f_frsize // (1024 * 1024), "needed_mb": size // (1024 * 1024)},
        )
        return None
    return live_dir

"""Data of saved instances: the head's public header, its sealed body, and what
describe_instance returns. Pure pydantic (no cryptography): the sealing lives in
head.py, codec.py and pack.py.
"""

from __future__ import annotations

import itertools
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exec_sandbox.instance.ids import COMMIT_ID_PATTERN, INSTANCE_ID_PATTERN
from exec_sandbox.models import InstanceRef
from exec_sandbox.vm_spec import VmSpec  # noqa: TC001 - pydantic resolves field types at runtime

RECENT_COMMITS = 16
"""Commits a head remembers, newest first: how far back a saver can still find
its own commit under newer ones."""

STREAM_VMSTATE = 0
"""Migration file: RAM at fixed offsets (mapped-ram), then device state."""
STREAM_HOME = 1
"""Home disk (/home/user, ext4) as of the save point."""

_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_STRICT = ConfigDict(frozen=True, extra="forbid", strict=True)


class RecentCommit(BaseModel):
    """One recent save of an instance: its generation and commit id."""

    model_config = _STRICT

    generation: int = Field(ge=1)
    """Generation this commit made (1 for the first save)."""
    commit_id: str = Field(pattern=COMMIT_ID_PATTERN)
    """Id of the save attempt; also in error.context["commit_id"] of an unclear save."""


class HeadHeader(BaseModel):
    """Public part of a head. Not authenticated until the head body opens
    (`head.open_head` or `head.authenticate`)."""

    model_config = _STRICT

    instance_id: str = Field(pattern=INSTANCE_ID_PATTERN)
    generation: int = Field(ge=1)
    commit_id: str = Field(pattern=COMMIT_ID_PATTERN)
    recent_commits: tuple[RecentCommit, ...] = Field(min_length=1, max_length=RECENT_COMMITS)
    """Newest first; the first entry is this head's own commit."""
    key_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Id of the keyring key that wraps the data key."""
    wrapped_data_key: str = Field(pattern=r"^[0-9a-f]{80}$")
    """AES-KW wrap of the 32-byte data key, hex."""

    @model_validator(mode="after")
    def _check_recent_commits(self) -> Self:
        first = self.recent_commits[0]
        if (first.generation, first.commit_id) != (self.generation, self.commit_id):
            raise ValueError("recent_commits must start with this head's commit")
        generations = [c.generation for c in self.recent_commits]
        if any(newer <= older for newer, older in itertools.pairwise(generations)):
            raise ValueError("recent_commits generations must decrease")
        return self


class HostFacts(BaseModel):
    """The host that saved the instance. A load on a host with another arch, accelerator,
    CPU vendor, guest CPU model or exec-sandbox release is refused before any download
    (ROUTE). Images are matched by digest (a worker without them downloads them from the
    store). qemu_version is for diagnostics: the load lets QEMU decide."""

    model_config = _STRICT

    arch: str
    """x86_64 or aarch64."""
    accelerator: str
    """kvm, hvf or tcg: a worker that falls back to tcg cannot load an instance saved with kvm."""
    cpu_vendor: str | None
    """x86 vendor the CPU model pins; None on ARM."""
    cpu_model: str = Field(min_length=1)
    """-cpu value of the saved VM's QEMU (qemu_cmd.guest_cpu_model): a restore needs the same
    CPU features. On x86 it holds +tsc-deadline when the host has that feature (the VM then
    runs with pit/pic/rtc off)."""
    qemu_version: str
    base_image: str = Field(pattern=_DIGEST_PATTERN)
    """SHA-256 of the base image: the guest's page cache refers to its bytes."""
    packages_image: str | None = Field(pattern=_DIGEST_PATTERN)
    """SHA-256 of the L2 packages image, None without packages."""
    release: str = Field(min_length=1)
    """exec-sandbox release that saved it: the guest agent in the saved RAM is of this release."""


class ChainEntry(BaseModel):
    """One body object of the chain: a full copy, or a pack on top of the entries before it."""

    model_config = _STRICT

    commit_id: str = Field(pattern=COMMIT_ID_PATTERN)
    kind: Literal["full", "pack"]
    size: int = Field(ge=0)
    """Bytes of the body object."""
    stream_sizes: tuple[int, int]
    """Byte sizes of (STREAM_VMSTATE, STREAM_HOME) at this commit: a load cuts or
    extends the streams to them after applying this object."""
    etag: str = Field(min_length=1, max_length=1024)
    """ETag the store gave the upload: a save compares it with a listing to find
    an object that was replaced."""


class HeadBody(BaseModel):
    """Sealed body of a head."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: VmSpec
    host: HostFacts
    chain: tuple[ChainEntry, ...] = Field(min_length=1)
    """Oldest first: one full copy, then packs."""

    @model_validator(mode="after")
    def _check_chain(self) -> Self:
        if self.chain[0].kind != "full" or any(entry.kind != "pack" for entry in self.chain[1:]):
            raise ValueError("a chain is one full copy followed by packs")
        return self


class InstanceInfo(InstanceRef):
    """What the store holds for one instance (Scheduler.describe_instance).

    The head is the small object in the store that names the current state of an
    instance. With the instance's key (`verified` True), every field comes from an
    authenticated head. Without it, `spec` and `host` are None and the other
    fields come from the public part of the head, which nothing authenticates. A
    head that fails authentication raises InstanceGoneError (INSTANCE_INTEGRITY); it
    never gives verified=False.
    """

    recent_commits: tuple[RecentCommit, ...] = Field(
        description="Newest first (at most 16): a save whose outcome was unknown was stored if this list "
        'holds its error.context["commit_id"]'
    )
    key_id: str = Field(description="Id of the keyring key that protects this instance (for key retirement)")
    verified: bool = Field(
        description="True: this keyring opened and authenticated the head. False: this keyring lacks the key"
    )
    spec: VmSpec | None = Field(description="What the VM is made of (None unless verified)")
    host: HostFacts | None = Field(
        description="Host it was saved on: arch, accelerator, CPU vendor, guest CPU model, image digests, "
        "exec-sandbox release (None unless verified)"
    )

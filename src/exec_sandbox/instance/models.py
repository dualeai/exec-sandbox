"""Data of saved instances: the head's public header, its sealed body, and what
describe_instance returns. Pure pydantic (no cryptography): the sealing lives in
head.py, codec.py and pack.py.
"""

from __future__ import annotations

import itertools
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from exec_sandbox.instance.ids import COMMIT_ID_PATTERN, INSTANCE_ID_PATTERN
from exec_sandbox.models import InstanceRef
from exec_sandbox.vm_spec import VmSpec  # noqa: TC001 - pydantic resolves field types at runtime

BLOCK_BYTES = 64 * 1024
"""Bytes of one stream block: a full copy and a pack seal each block as one record. Each
chain entry states it (ChainEntry.block_bytes): a release with another block size cannot
place the records and gives RouteError (another release) before the download."""

RECENT_COMMITS = 16
"""Commits a head remembers, newest first: how far back a saver can still find
its own commit under newer ones."""

STREAM_VMSTATE = 0
"""Migration file: RAM at fixed offsets (mapped-ram), then device state."""
STREAM_HOME = 1
"""Home disk (/home/user, ext4) as of the save point."""

InstanceAccess = Literal["full", "no_key", "other_release"]
"""How much of a head this worker could read (InstanceInfo.access)."""

_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_STRICT = ConfigDict(frozen=True, extra="forbid", strict=True)


class RecentCommit(BaseModel):
    """One recent save of an instance: its generation and commit id."""

    model_config = _STRICT

    generation: int = Field(ge=1)
    """Generation this commit made (1 for the first save)."""
    commit_id: str = Field(pattern=COMMIT_ID_PATTERN)
    """Id of the save attempt; also in error.context["commit_id"] of an unclear save."""


class HeadEnvelope(BaseModel):
    """The header fields a reader needs before it can authenticate a head: the same
    names and forms in every release. The other header fields are parsed only after
    the head authenticates, so a release that adds or changes one gets RouteError on
    the others, never INSTANCE_INTEGRITY."""

    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)

    instance_id: str = Field(pattern=INSTANCE_ID_PATTERN)
    key_id: str = Field(pattern=_DIGEST_PATTERN)
    wrapped_data_key: str = Field(pattern=r"^[0-9a-f]{80}$")


class HeadSummary(BaseModel):
    """The header fields describe_instance shows, read without the rest of the header
    (other fields are ignored): a header that another release wrote, with fields this
    release does not know, still shows its generation, recent commits and key."""

    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)

    instance_id: str = Field(pattern=INSTANCE_ID_PATTERN)
    generation: int = Field(ge=1)
    recent_commits: tuple[RecentCommit, ...] = Field(min_length=1)
    key_id: str = Field(pattern=_DIGEST_PATTERN)


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
    """The host of the last save, recomputed at each save. A load on a host with another
    arch, accelerator, guest CPU model or CPU identity, or whose QEMU lacks the machine type,
    is refused before any download (ROUTE). qemu_version: QEMU decides at the restore; after
    a refusal it names the QEMU to route to (that version or newer). release: a diagnostic.

    A new fact goes here when it is the saving host's (a later save on another host
    replaces it), in GuestFacts when it comes from the guest's boot (no save changes it)."""

    model_config = _STRICT

    arch: str
    """x86_64 or aarch64."""
    accelerator: str
    """kvm, hvf or tcg: a worker that falls back to tcg cannot load an instance saved with kvm."""
    cpu_model: str = Field(min_length=1)
    """-cpu value of the saved VM's QEMU (qemu_cmd.guest_cpu_model): a restore needs the same
    CPU features. On x86 it holds +tsc-deadline when the host has that feature (the VM then
    runs with pit/pic/rtc off)."""
    cpu_id: str | None = Field(min_length=1)
    """The host CPU the guest sees under -cpu host (qemu_cmd.guest_cpu_id: ARM64 KVM and HVF);
    None where the guest CPU is a fixed model (x86, ARM64 TCG)."""
    qemu_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    """QEMU of the last save, MAJOR.MINOR.PATCH."""
    machine_type: str = Field(min_length=1)
    """QEMU machine type of the saved VM as QEMU resolved it (virt-11.1, microvm). A load
    starts QEMU with it, so a newer QEMU applies the compat properties of that type
    (versioned types stay 6 years); a QEMU without it cannot load the state (ROUTE)."""
    release: str = Field(min_length=1)
    """exec-sandbox release of the last save (a diagnostic: a load does not check it)."""
    device_layout: str = Field(min_length=1)
    """The guest-visible layout of the saved VM's QEMU command (qemu_cmd.device_layout): one
    line for each option with layout properties (the board, the RAM backend, each drive and
    device) in command order, then the migration capabilities. A load compares it with the
    command it would run, before any download: another layout loads silently wrong or not at
    all (ROUTE: another release)."""


class GuestFacts(BaseModel):
    """What the guest got at its cold boot, the files it booted on and the rate of its
    clock's counter: every save carries them unchanged (a reused VM runs the RAM of that
    boot, on the same disks). The images are
    matched by digest at a load (a worker without them downloads them from the store); the
    kernel and initramfs only name what runs (a load boots nothing), for a cleanup or a
    vulnerability audit by digest."""

    model_config = _STRICT

    base_image: str = Field(pattern=_DIGEST_PATTERN)
    """SHA-256 of the base image: the guest's page cache refers to its bytes."""
    packages_image: str | None = Field(pattern=_DIGEST_PATTERN)
    """SHA-256 of the L2 packages image, None without packages."""
    kernel: str = Field(pattern=_DIGEST_PATTERN)
    """SHA-256 of the kernel the guest booted (it runs in the saved RAM)."""
    initramfs: str = Field(pattern=_DIGEST_PATTERN)
    """SHA-256 of the initramfs the guest booted (its init and agent run in the saved RAM)."""
    counter_hz: int | None = Field(gt=0)
    """Rate in Hz of the counter the guest's clock runs on (x86 TSC, ARM generic timer), as the
    host of its cold boot gave it (qemu_cmd.CounterRate): the guest calibrated its clock
    against it, so a load needs a worker with this rate or with TSC scaling. None: that host
    had no exact source (no check)."""


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
    block_bytes: int
    """BLOCK_BYTES of the release that wrote the object: its records are blocks of this size.
    Another size does not parse here (another release wrote it)."""

    @field_validator("block_bytes")
    @classmethod
    def _this_block_size(cls, value: int) -> int:
        if value != BLOCK_BYTES:
            raise ValueError(f"blocks of {value} bytes; this release reads blocks of {BLOCK_BYTES}")
        return value


class HeadBody(BaseModel):
    """Sealed body of a head."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: VmSpec
    host: HostFacts
    guest: GuestFacts
    egress_domains: tuple[str, ...]
    """Domains the VM could reach with the network on, as resolved at the save (the spec's
    allowed_domains, None: the language's package registries; empty: all): a load applies
    this list, so a release whose default list differs does not change what a saved VM may
    reach. The spec stays as the user gave it."""
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
    instance. `access` says how much of it this worker could read:
    - "full": this keyring opened the head and this release read it; every field is
      authenticated.
    - "other_release": the head authenticated, but this release cannot read its body
      (another release wrote it): `spec`, `host`, `guest` and `egress_domains` are None;
      `writer_release` names that release when the body does (else None). The data is
      intact.
    - "no_key": this keyring lacks `key_id`: `spec`, `host`, `guest`, `egress_domains`
      and `writer_release` are None, and the other fields come from the public part of the
      head, which nothing authenticates.
    A head that fails authentication raises InstanceGoneError (INSTANCE_INTEGRITY).
    """

    recent_commits: tuple[RecentCommit, ...] = Field(
        description="Newest first (at most 16): a save whose outcome was unknown was stored if this list "
        'holds its error.context["commit_id"]'
    )
    key_id: str = Field(description="Id of the keyring key that protects this instance (for key retirement)")
    access: InstanceAccess = Field(
        description='"full": authenticated and read; "other_release": authenticated, but another release wrote '
        'the body; "no_key": this keyring lacks the key, nothing is authenticated'
    )
    writer_release: str | None = Field(
        description="exec-sandbox release that wrote the head (the last save); None when access is no_key, "
        "or when access is other_release and the body names no release"
    )
    spec: VmSpec | None = Field(description='What the VM is made of (None unless access is "full")')
    host: HostFacts | None = Field(
        description="Host of the last save: arch, accelerator, guest CPU model, host CPU identity, QEMU version "
        'and machine type, release, device layout (None unless access is "full")'
    )
    guest: GuestFacts | None = Field(
        description="What the guest got at its first boot: digests of its base and packages images, kernel, "
        'initramfs, and the rate of its clock counter (None unless access is "full")'
    )
    egress_domains: tuple[str, ...] | None = Field(
        description="Domains the VM may reach with the network on, as resolved at its last save and applied "
        'at each load (empty: all; None unless access is "full")'
    )

    @model_validator(mode="after")
    def _check_access(self) -> Self:
        full = self.access == "full"
        if any((field is not None) != full for field in (self.spec, self.host, self.guest, self.egress_domains)):
            raise ValueError('spec, host, guest and egress_domains are set exactly when access is "full"')
        if (self.access == "full" and self.writer_release is None) or (
            self.access == "no_key" and self.writer_release is not None
        ):
            raise ValueError('writer_release is set when access is "full" and None when access is "no_key"')
        return self

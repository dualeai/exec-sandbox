"""VM type definitions: states, acceleration types, transition rules, and what a guest got at
its boot."""

from dataclasses import dataclass
from enum import Enum


class VmState(Enum):
    """VM lifecycle states."""

    CREATING = "creating"
    BOOTING = "booting"
    READY = "ready"
    EXECUTING = "executing"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


@dataclass(frozen=True, slots=True, kw_only=True)
class GuestBoot:
    """What a guest got at its cold boot and keeps in every saved state (a restore runs the
    RAM of that boot): a restore passes the saved values (GuestFacts)."""

    kernel: str | None
    """SHA-256 of the kernel it booted; None: unknown (the file was missing at its start: no
    save records it)."""
    initramfs: str | None
    """SHA-256 of the initramfs it booted; None as for `kernel`."""
    counter_hz: int | None
    """Rate in Hz of the counter its clock runs on, as its boot host gave it (qemu_cmd.CounterRate);
    None: that host had no exact source."""


class AccelType(Enum):
    """QEMU acceleration type."""

    KVM = "kvm"  # Linux hardware virtualization
    HVF = "hvf"  # macOS hardware virtualization
    TCG = "tcg"  # Software emulation (slow, but works everywhere)


# Valid state transitions for VM lifecycle
VALID_STATE_TRANSITIONS: dict[VmState, set[VmState]] = {
    VmState.CREATING: {VmState.BOOTING, VmState.DESTROYING},
    VmState.BOOTING: {VmState.READY, VmState.DESTROYING},
    VmState.READY: {VmState.EXECUTING, VmState.DESTROYING},
    VmState.EXECUTING: {VmState.READY, VmState.DESTROYING},
    VmState.DESTROYING: {VmState.DESTROYED},
    VmState.DESTROYED: set(),  # Terminal state - no transitions allowed
}

"""VM type definitions: states, acceleration types, and transition rules."""

from enum import Enum


class VmState(Enum):
    """VM lifecycle states."""

    CREATING = "creating"
    BOOTING = "booting"
    READY = "ready"
    EXECUTING = "executing"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


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

# Validate all states have transition rules defined
if set(VmState) != set(VALID_STATE_TRANSITIONS.keys()):
    _missing = set(VmState) - set(VALID_STATE_TRANSITIONS.keys())
    raise RuntimeError(f"Missing states in transition table: {_missing}")

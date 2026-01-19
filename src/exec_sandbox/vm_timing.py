"""VM timing instrumentation dataclass.

Provides a structured way to track and report timing information for VM lifecycle events.
Used for performance monitoring, debugging, and optimization.
"""

from dataclasses import asdict, dataclass


@dataclass
class VmTiming:
    """Timing instrumentation for VM lifecycle events.

    All values are in milliseconds. None indicates the timing was not recorded
    (e.g., feature was disabled or error occurred before that phase).

    High-level timing:
        setup_ms: Total time for resource setup (overlay, cgroup, etc.)
        boot_ms: Total time for VM boot (kernel + guest agent init)

    Granular setup timing:
        overlay_ms: Time to acquire overlay from pool or create new

    Granular boot timing:
        qemu_cmd_build_ms: Time to build QEMU command line
        gvproxy_start_ms: Time to start gvproxy (0 if network disabled)
        qemu_fork_ms: Time for QEMU process fork/exec
        guest_wait_ms: Time waiting for guest agent ready signal

    Example usage:
        timing = VmTiming(setup_ms=50, boot_ms=250, overlay_ms=10, ...)
        print(timing.to_dict())  # Only includes non-None values
    """

    # High-level timing (aggregates)
    setup_ms: int | None = None
    boot_ms: int | None = None

    # Granular setup timing
    overlay_ms: int | None = None

    # Granular boot timing
    qemu_cmd_build_ms: int | None = None
    gvproxy_start_ms: int | None = None
    qemu_fork_ms: int | None = None
    guest_wait_ms: int | None = None

    # Retry tracking (for CPU contention resilience)
    boot_retries: int | None = None  # 0 = succeeded first try, 1+ = number of retries

    def to_dict(self) -> dict[str, int]:
        """Convert to dict with only non-None values.

        Returns:
            Dictionary mapping timing names to millisecond values,
            excluding any timing that was not recorded (None).

        Example:
            >>> timing = VmTiming(setup_ms=50, boot_ms=None)
            >>> timing.to_dict()
            {'setup_ms': 50}
        """
        return {k: v for k, v in asdict(self).items() if v is not None}

    @property
    def total_ms(self) -> int | None:
        """Total time (setup + boot) if both are available.

        Returns:
            Total milliseconds if both setup_ms and boot_ms are set, None otherwise.
        """
        if self.setup_ms is not None and self.boot_ms is not None:
            return self.setup_ms + self.boot_ms
        return None

"""VM timing instrumentation dataclass.

Provides a structured way to track and report timing information for VM lifecycle events.
Used for performance monitoring, debugging, and optimization.
"""

from dataclasses import dataclass


@dataclass
class VmTiming:
    """Timing instrumentation for VM lifecycle events.

    All values are in milliseconds. None indicates the timing was not recorded
    (e.g., feature was disabled or error occurred before that phase).
    VmManager sets them (create_vm; restore_vm sets only setup_ms and boot_ms).

    High-level timing:
        setup_ms: Total time for resource setup (working folder, overlay, cgroup, channel);
            for a restore (L1 or saved instance), the whole restore
        boot_ms: Time from the end of setup to a ready guest agent (QEMU command build,
            gvproxy start, QEMU spawn, kernel and guest agent init); 0 for a restore

    Granular setup timing:
        overlay_ms: Time to create the overlay (QemuStorageDaemon.create_overlay: a clone
            of the base image's template)

    Granular boot timing:
        qemu_cmd_build_ms: Time to build QEMU command line
        gvproxy_start_ms: Time to start the gvproxy process (0 if network disabled); its ready
            wait overlaps the boot (in guest_wait_ms)
        qemu_fork_ms: Time for QEMU process fork/exec
        guest_wait_ms: Time waiting for guest agent ready signal

    Retries:
        boot_retries: Boot attempts that failed before the one that succeeded (create_vm)
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

"""Constants for exec-sandbox configuration and limits."""

import enum
from dataclasses import dataclass
from typing import Final

# ============================================================================
# Guest Agent Error Types (wire format vocabulary)
# ============================================================================


class GuestErrorType(enum.StrEnum):
    """Domain-specific error types from the guest agent wire protocol.

    Maps 1:1 to Rust ErrorType enum in guest-agent/src/error.rs.
    """

    ENV_VAR = "env_var_error"
    CODE = "code_error"
    PATH = "path_error"
    PACKAGE = "package_error"
    IO = "io_error"
    TIMEOUT = "timeout_error"
    EXECUTION = "execution_error"
    REQUEST = "request_error"
    PROTOCOL = "protocol_error"
    OUTPUT_LIMIT = "output_limit_error"


# ============================================================================
# VM Memory and Resource Defaults
# ============================================================================

DEFAULT_MEMORY_MB: Final[int] = 192
"""Default guest VM memory allocation in MB.

Bun needs ~42MB RSS plus the kernel overhead: 192MB gives enough headroom at a
low per-VM cost."""

MIN_MEMORY_MB: Final[int] = 128
"""Minimum guest VM memory in MB.

NOTE: JavaScript (Bun) requires significantly more memory than Python.
Bun RSS ~42MB vs CPython ~19MB. At 128MB VMs (~40MB free RAM), Bun
swap-thrashes via zram causing 40s+ latency for trivial operations.
Recommended minimums: Python 128MB, JavaScript 160MB+, ideal 192MB+.
The --smol flag in guest-agent mitigates this but doesn't eliminate it.
"""

# ============================================================================
# Execution Timeouts
# ============================================================================

DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
"""Default code execution timeout in seconds."""

MAX_TIMEOUT_SECONDS: Final[int] = 300
"""Maximum code execution timeout in seconds (5 minutes)."""

VM_BOOT_TIMEOUT_SECONDS: Final[int] = 45
"""VM boot timeout in seconds (guest agent ready check).

Sized for CI nested virtualization (AMD nested KVM: 50-90% CPU degradation per
Cloud Hypervisor issue #4827). Bare-metal boots complete in <1s; the timeout is
for the full boot + guest-agent ready sequence under worst-case contention."""

VM_BOOT_MAX_RETRIES: Final[int] = 4
"""Maximum attempts for a VM boot (CPU contention resilience).

Four attempts give a transient overlay daemon error (connection refused at
startup) time to clear."""

VM_BOOT_RETRY_MIN_SECONDS: Final[float] = 0.1
"""Minimum backoff between VM boot retries (100ms base).

Under CI CPU contention the host scheduler needs time to reschedule the QEMU
process: 20ms retries are too tight."""

VM_BOOT_RETRY_MAX_SECONDS: Final[float] = 2.0
"""Maximum backoff between VM boot retries (2s cap with jitter).

Lets the overlay daemon recover from transient connection-refused errors
(a daemon restart takes ~100-500ms)."""


@dataclass(frozen=True, slots=True)
class RetryProfile:
    """Bundled retry parameters for VM boot.

    Pre-built profiles tune retry aggressiveness to caller context:
    - User-facing: generous retries for best success rate
    - Background: fewer retries, shorter timeouts for sacrificial VMs
    """

    max_attempts: int
    boot_timeout_seconds: int
    retry_min_seconds: float
    retry_max_seconds: float


RETRY_USER_FACING: Final[RetryProfile] = RetryProfile(
    max_attempts=VM_BOOT_MAX_RETRIES,
    boot_timeout_seconds=VM_BOOT_TIMEOUT_SECONDS,
    retry_min_seconds=VM_BOOT_RETRY_MIN_SECONDS,
    retry_max_seconds=VM_BOOT_RETRY_MAX_SECONDS,
)
"""Default retry profile for user-facing VMs (scheduler)."""

RETRY_BACKGROUND: Final[RetryProfile] = RetryProfile(
    max_attempts=2,
    boot_timeout_seconds=30,
    retry_min_seconds=VM_BOOT_RETRY_MIN_SECONDS,
    retry_max_seconds=VM_BOOT_RETRY_MAX_SECONDS,
)
"""Retry profile for background/disposable VMs (warm pool, snapshot creation, L1 saves).
Fewer retries and shorter timeout — these VMs are sacrificial."""

GUEST_CONNECT_TIMEOUT_SECONDS: Final[int] = 5
"""Timeout for connecting to guest agent."""

GUEST_REQUEST_TIMEOUT_SECONDS: Final[int] = 5
"""Default timeout for guest agent request/response (ping, etc.)."""

WARM_REPL_TIMEOUT_SECONDS: Final[int] = 120
"""Timeout for WarmReplRequest (REPL process startup in guest).

Python REPL takes 11+ seconds on HVF (Alpine CPython cold start), and under
resource contention (background save runs concurrently with user execution)
it can take 2-3x longer. 120s is generous since this is used for background
L1 saves and warm pool pre-warm — latency doesn't matter, only success."""

GUEST_RECONNECT_PROBE_TIMEOUTS: Final[tuple[float, ...]] = (0.05, 0.1, 0.25, 0.5, 1.0, 1.0)
"""Timeout of each ping attempt that checks the guest after a reconnection (an
L1 restore, or a channel that was closed).

Each attempt sends a new ping on the same connection: the guest keeps its ports
open, so a ping is held until the guest reads it, never lost. The waits grow:
a pong comes in 1-3 ms on HVF and 13-17 ms on nested KVM (20-40 ms with 16
restores at once), so the first wait ends the probe; a slow guest (TCG, a
loaded host) gets the longer later waits. The sum (2.9 s) is the budget before
the restore counts as failed."""

GUEST_RECONNECT_PROBE_DELAY: Final[float] = 0.05
"""Delay before a new connection when the probe's connection failed (50ms)."""

GUEST_PORT_IDS: Final[tuple[str, str]] = ("guest-cmd", "guest-event")
"""QEMU ids of the command and event ports of the guest agent (virtserialport). A port
with an id makes QEMU send VSERPORT_CHANGE when the guest opens or closes it."""

SIGKILL_EXIT_CODE: Final[int] = 128 + 9
"""Exit code reported when a process died from SIGKILL (guard, OOM killer, or
user code exiting 137). Terminal for sessions — see Session.exec."""

EXECUTION_TIMEOUT_MARGIN_SECONDS: Final[int] = 10
"""Hard timeout margin above soft timeout (host watchdog protection).
Accounts for:
- Guest graceful termination grace period (5s SIGTERM→SIGKILL)
- JSON serialization, transport, scheduling, and clock skew (~3s)
- First-execution slack and final safety buffer (~2s)
Zram readiness is terminal before the guest listener starts, so it can never
delay a request. Network readiness gates only the first execution and usually
resolves in well under a second; a pathologically slow (multi-second) setup on
that first execution may trip this watchdog. The VM is then discarded and the
caller sees OutcomeUnknownError (after dispatch) or a boot
timeout (before dispatch) — never a wrong result. That rare case is accepted
instead of delaying hung-guest detection by 8s on every execution. The
package-install path, which is always a first op on a network VM, adds its
own gate allowance (see disk_snapshot_manager).
This is a hard infrastructure backstop; user-code timeout semantics are
unchanged."""

SNAPSHOT_CREATION_MIN_MEMORY_MB: Final[int] = 320
"""Minimum guest memory for snapshot creation VMs (package installation).

Snapshot creation runs uv/bun (large binaries) inside the guest. Under TCG,
the cold-start JIT translation of uv (~30MB Rust binary) plus TLS crypto drives
RSS above 192MB, triggering zram compression which runs in emulated code (5-8x
slower). 320MB provides 128MB headroom above the TCG pressure threshold. On
KVM/HVF, 192MB suffices but the minimum is kept conservative for TCG CI
environments."""

PACKAGE_INSTALL_TIMEOUT_SECONDS: Final[int] = 300
"""Timeout for package installation in guest VM.

Sent to the guest agent via InstallPackagesRequest.timeout field.
The guest enforces this as the tokio::time::timeout for the package manager process.
Also used host-side to compute hard_timeout (+ EXECUTION_TIMEOUT_MARGIN_SECONDS + 8s first-op network gate allowance, see disk_snapshot_manager)."""

PACKAGE_INSTALL_MAX_RETRIES: Final[int] = 3
"""Maximum retry attempts for transient network errors during package install."""

PACKAGE_INSTALL_RETRY_MIN_SECONDS: Final[float] = 1.0
"""Minimum backoff between package install retries (1s base).

Longer than VM boot retry (0.1s) because network issues persist longer than CPU contention."""

PACKAGE_INSTALL_RETRY_MAX_SECONDS: Final[float] = 8.0
"""Maximum backoff between package install retries (8s cap with jitter).

Longer than VM boot retry (2s) to allow transient DNS/routing issues to resolve."""

# ============================================================================
# Code Execution Limits
# ============================================================================

MAX_CODE_SIZE: Final[int] = 1024 * 1024  # 1 MiB
"""Maximum size in bytes for code input."""

MAX_ENV_VARS: Final[int] = 100
"""Maximum number of environment variables allowed."""

MAX_ENV_VAR_NAME_LENGTH: Final[int] = 256
"""Maximum length for environment variable names."""

MAX_ENV_VAR_VALUE_LENGTH: Final[int] = 4096
"""Maximum length for environment variable values."""

# Control characters forbidden in env var names/values (security risk)
# Allows: tab (0x09), printable ASCII (0x20-0x7E), UTF-8 multibyte (0x80+)
# Forbids: NUL, C0 controls (except tab), DEL
ENV_VAR_FORBIDDEN_CONTROL_CHARS: Final[frozenset[int]] = frozenset(
    list(range(0x09))  # NUL, SOH, STX, ETX, EOT, ENQ, ACK, BEL, BS
    + list(range(0x0A, 0x20))  # LF, VT, FF, CR, SO through US (includes ESC at 0x1B)
    + [0x7F]  # DEL
)
"""Forbidden control characters in env var names/values (terminal escape injection prevention)."""


# ============================================================================
# File I/O
# ============================================================================

MAX_FILE_SIZE_BYTES: Final[int] = 500 * 1024 * 1024  # 500MB
"""Maximum file size for read/write operations.

With streaming zstd-compressed chunks, the host holds at most about 13.5 MiB
of encoded chunks per write in progress (HOST_WRITE_QUEUE_DEPTH), whatever the
file size. The practical ceiling is the free space of the guest home disk
(/home/user, VmSpec.disk_mb), not RAM.
"""

HOST_WRITE_QUEUE_DEPTH: Final[int] = 64
"""Frames the host write queue holds (guest_channel.UnixSocketChannel). With
HOST_WRITE_BATCH_FRAMES it bounds a write_file in progress: (HOST_WRITE_QUEUE_DEPTH
+ HOST_WRITE_BATCH_FRAMES + 1) frames of ~175 KB (a 128 KiB chunk that does not
compress, base64 in JSON), about 13.5 MiB, whatever the file size."""

HOST_WRITE_BATCH_FRAMES: Final[int] = 16
"""Frames the write worker writes before one drain(). They stay in the transport
until sent, with the partly sent frame of the batch before."""

MAX_FILE_PATH_LENGTH: Final[int] = 4096
"""Maximum relative file path length in bytes (POSIX PATH_MAX).

Individual filename components are limited to 255 bytes (NAME_MAX) by the
guest agent's validate_file_path(). This constant covers the full relative path.
"""

FILE_IO_TIMEOUT_SECONDS: Final[int] = 120
"""Timeout for file I/O operations in seconds."""

# ============================================================================
# Outbound Domain Filtering
# ============================================================================
# gvproxy filters outbound DNS queries and TLS SNI with OutboundAllow patterns
# (dns_filter, gvproxy.start_gvproxy)

PYTHON_PACKAGE_DOMAINS: Final[list[str]] = [
    "pypi.org",
    "files.pythonhosted.org",
]
"""Default domain whitelist for Python package installation."""

NPM_PACKAGE_DOMAINS: Final[list[str]] = [
    "registry.npmjs.org",
]
"""Default domain whitelist for JavaScript/Node package installation."""

# ============================================================================
# System Limits
# ============================================================================

DEADLINE_DIAGNOSTICS_TIMEOUT_SECONDS: Final[float] = 1.0
"""Bound on the QMP query-status of a VM whose host deadline expired (a failure
path only: one connection and two QMP round trips, a few ms for a healthy QEMU)."""

DEADLINE_LOG_CONSOLE_CHARS: Final[int] = 4000
"""End of the console ring buffer that a host-deadline log line shows (the error
context holds the whole ring buffer)."""

CONSOLE_RING_LINES: Final[int] = 500
"""In-memory ring buffer size (lines) for QEMU console output.

Kernel loglevel=7 can emit 300-600 lines before init starts.
500 keeps the diagnostic tail intact for boot failure analysis."""

# ============================================================================
# Disk I/O Performance Limits
# ============================================================================

DISK_BPS_LIMIT: Final[int] = 50 * 1024 * 1024  # 50 MB/s
"""Sustained disk bandwidth limit in bytes/second (prevent noisy neighbor)."""

DISK_BPS_BURST: Final[int] = 100 * 1024 * 1024  # 100 MB/s
"""Burst disk bandwidth limit in bytes/second (package downloads)."""

DISK_IOPS_LIMIT: Final[int] = 1000
"""Sustained IOPS limit (typical code execution workload)."""

DISK_IOPS_BURST: Final[int] = 2000
"""Burst IOPS limit (npm install, pip install)."""

VIRTIO_QUEUE_SIZE: Final[int] = 64
"""Queue depth (entries per virtqueue) of each virtio-blk device (qemu_cmd: queue-size).

Each entry costs ~90 bytes of host memory (16-byte descriptor + 2-byte avail
ring entry + 8-byte used ring entry + ~64 bytes host metadata). At 64 entries a
virtqueue costs ~5.6KB. Each virtio-blk device (base image, L2 snapshot, home
disk) has one queue (num-queues=1).

QEMU's default is 256. Sandbox I/O is sequential and low-throughput: 64 entries
are enough, with no measurable wall-time difference in cold-start benchmarks
(the bottleneck is the Mach kernel / KVM, not the queue depth)."""

# ============================================================================
# Warm VM Pool
# ============================================================================

WARM_POOL_REPLENISH_CONCURRENCY_RATIO: Final[float] = 0.5
"""Max concurrent replenish boots as ratio of pool_size (50% = 2-3 concurrent for pool_size=5)."""

WARM_POOL_TENANT_ID: Final[str] = "warm-pool"
"""Placeholder tenant ID for warm pool VMs."""

L1_SAVE_TENANT_ID: Final[str] = "l1-cache"
"""Tenant ID for sacrificial VMs created during L1 background saves."""

SCHEDULER_TENANT_ID: Final[str] = "exec-sandbox"
"""Tenant ID for user-facing VMs created by the scheduler."""

WARM_POOL_HEALTH_CHECK_INTERVAL: Final[int] = 15
"""Health check interval for warm VMs in seconds."""

WARM_POOL_HEALTH_CHECK_MAX_RETRIES: Final[int] = 3
"""Maximum retry attempts before declaring VM unhealthy (matches K8s failureThreshold)."""

WARM_POOL_HEALTH_CHECK_RETRY_MIN_SECONDS: Final[float] = 0.1
"""Minimum backoff between health check retries."""

WARM_POOL_HEALTH_CHECK_RETRY_MAX_SECONDS: Final[float] = 2.0
"""Maximum backoff between health check retries."""

# ============================================================================
# Memory Snapshot (L1 Cache)
# ============================================================================

MIGRATION_TIMEOUT_BASE_SECONDS: Final[float] = 30.0
"""Fixed part of the save/restore timeout (migrate or migrate-incoming + poll)."""

MIGRATION_MIN_THROUGHPUT_MB_PER_SECOND: Final[float] = 100.0
"""Slowest disk rate the save/restore timeout allows for the whole guest RAM."""

MEMORY_SNAPSHOT_QMP_TIMEOUT_SECONDS: Final[float] = 10.0
"""Timeout for individual QMP commands during migration."""

MEMORY_SNAPSHOT_POLL_INTERVAL_SECONDS: Final[float] = 0.005
"""Interval between query-migrate polls (5ms)."""

MEMORY_SNAPSHOT_QMP_SOCKET_TIMEOUT_SECONDS: Final[float] = 5.0
"""Timeout waiting for QMP socket to appear during L1 restore (sudo path only).

Only used when socket activation is unavailable (sudo wraps QEMU, closing
inherited fds). With socket activation, the socket is pre-created and no
polling is needed."""

QEMU_SANDBOX_POLICY: Final[str] = "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny"
"""The -sandbox (seccomp) value of every VM, and of the probe that checks this host
accepts it (system_probes.probe_qemu_sandbox_support): one value, so the probe checks
the filter VMs install."""

MEMORY_SNAPSHOT_MIN_QEMU_VERSION: Final[tuple[int, int, int]] = (11, 0, 0)
"""Minimum QEMU version for mapped-ram + multifd migration capabilities.

Floor is 11.0: it is the first release where mapped-ram coexists with
x-ignore-shared without a null-deref ("Bad address") crash, letting the template
and full-RAM paths share ONE unified capability list (no version fork). The
QEMU pin in versions.lock is above this floor, so it is free. Below the floor,
save and restore raise ConfigError MIGRATION_UNSUPPORTED (no legacy fallback). Changing
capabilities requires bumping MEMORY_SNAPSHOT_FORMAT_VERSION."""

MEMORY_SNAPSHOT_FORMAT_VERSION: Final[int] = 5
"""Migration format version. Bump when changing QMP capabilities or migration
protocol to invalidate stale L1 cache entries. History:
  v1: Plain streaming format (no mapped-ram)
  v2: mapped-ram + multifd (QEMU >= 9.0)
  v3: x-ignore-shared VM templating — device-state-only vmstate + separate
      RAM file for COW sharing across VMs (QEMU >= 9.0)
  v4: unified capability list — template path now also enables mapped-ram +
      multifd alongside x-ignore-shared (QEMU >= 11.0 fixes the coexistence
      crash), removing the version-forked branch.
  v5: L2 snapshot drive (hd1) opened read-only outside the L2 build — the
      virtio-blk device now carries VIRTIO_BLK_F_RO, which v4 templates lack;
      x-ignore-shared state always sent explicitly; no legacy streaming saves;
      no virtio-balloon; home disk drive (hd2, serial=home) with a .home file
      per entry.

NOTE: the migration stream layout is QEMU-version- AND parameter-specific — a
QEMU upgrade or a change to multifd-channels / direct-io (both baked into the
mapped-ram file layout) requires restore-QEMU == save-QEMU. Any such change must
bump this version so stale, incompatible snapshots are not restored.
"""

# ============================================================================
# Overlay Pool
# ============================================================================

OVERLAY_POOL_FALLBACK_SIZE: Final[int] = 5
"""Overlay pool size when host resources are unknown (psutil unavailable).

Provides a reasonable default for pre-created overlays per base image
when the admission controller cannot determine host capacity."""

OVERLAY_POOL_SIZE_RATIO: Final[float] = 1.0
"""Overlay pool size as ratio of effective max VMs (100%).

Matches the max concurrent VM count, so a full admission cycle gets pool hits
(<1 ms). The pool fills in the background after start (84 overlays in ~125 ms on
an M-series Mac); until then acquire() creates overlays on demand (~0.3 ms once the
template of the base image exists, ~5 ms for the first).
With 24 max VMs, pool will have 24 pre-created overlays per base image
(~200KB each = ~5MB per image, negligible disk cost).
"""

OVERLAY_POOL_CREATION_CONCURRENCY: Final[int] = 2
"""Overlays the pool fill creates at once (the on-demand path of acquire() does not wait for it).

The first overlay of each base image version is QMP exchanges, which the daemon serves one at
a time (~5 ms); each later one is a clone of that template (~0.3 ms, no QMP). With 2 fill
workers, 3 base images x 28 overlays are ready ~125 ms after start on an M-series Mac
(822 ms when each overlay is made through QMP).
"""

OVERLAY_POOL_REPLENISH_INTERVAL_SECONDS: Final[float] = 0.5
"""Interval between replenishment checks."""

# ============================================================================
# QEMU Storage Daemon
# ============================================================================

QEMU_STORAGE_DAEMON_SOCKET_TIMEOUT_SECONDS: Final[float] = 5.0
"""Timeout for QMP socket operations."""

QEMU_STORAGE_DAEMON_STARTUP_TIMEOUT_SECONDS: Final[float] = 10.0
"""Timeout waiting for daemon to become ready."""

QEMU_STORAGE_DAEMON_JOB_TIMEOUT_SECONDS: Final[float] = 30.0
"""Timeout waiting for async blockdev-create jobs to complete."""

# ============================================================================
# Port Forwarding
# ============================================================================

MAX_EXPOSED_PORTS: Final[int] = 10
"""Maximum number of ports that can be exposed per VM."""

PORT_FORWARD_BIND_HOST: Final[str] = "127.0.0.1"
"""Host address to bind forwarded ports to (localhost only for security)."""

# ============================================================================
# File Transfer Streaming
# ============================================================================

FILE_TRANSFER_CHUNK_SIZE: Final[int] = 128 * 1024  # 128KB
"""Raw chunk size for streaming file transfers (before zstd compression).

128KB halves the frames of 64KB chunks (syscalls, JSON parses, base64
encodes and decodes). On-wire size after base64+JSON is ~175KB per frame.
"""

FILE_TRANSFER_ZSTD_LEVEL: Final[int] = 3
"""Zstd compression level for file transfers (1=fast, 22=max; 3=good balance)."""

# ============================================================================
# Resource Admission & Overcommit
# ============================================================================

DEFAULT_DISK_OVERCOMMIT_RATIO: Final[float] = 4.0
"""Home disk overcommit multiplier: gate 4 reserves 1/ratio (a quarter) of the part of each
home disk not written yet. Home disks are sparse and guests write little of them (a new
1 GB disk uses ~17 MB): a reserve of the whole size (1.0) makes free disk a hard VM count.
A guest that writes past the host's free space is paused by QEMU (werror=enospc) and its
operations fail with LOCAL_DISK; the floor (disk_free_floor_mb) keeps room for the host and
the caches. Above 64, an unwritten 1 GB disk would reserve less than a new disk writes."""

DEFAULT_MEMORY_OVERCOMMIT_RATIO: Final[float] = 5.0
"""Host-level memory overcommit multiplier applied to the total VM pool budget.

budget = host_total * (1 - reserve_ratio) * overcommit_ratio

This stretches the pool beyond physical capacity — safe because AI agent workloads
are bursty and mostly idle (avg memory utilization 18-22%).  At the optimal CPU=2x,
CPU is the binding constraint (~24 concurrent slots), so memory overcommit only
affects the theoretical budget ceiling, not actual concurrency or RSS.

The 5.0x default was determined by RBF surrogate optimization (thin-plate spline,
16 samples) over a 4x4 latency-frontier sweep (250 VMs per combo) on a 16-vCPU /
32GB EC2 instance with KVM + KSM (scripts/benchmark_optimizer.py):

  Predicted optimal: CPU_OC=2.0, MEM_OC=4.6 (efficiency=2227.9)
  Best sampled:      CPU_OC=2.0, MEM_OC=5.0 (efficiency=2223.3)
  → Grid already covers the optimum

CPU=2x MEM=5x achieves latency efficiency=2223 (10^6 / geomean of admission,
setup, exec p95) with admit_p95=3025ms, setup_p95=160ms, exec_p95=188ms at
3.4% RSS.  13/16 grid combos achieved 100% success (250 VMs each); CPU=12x
combos degraded under extreme contention (13-100% success, 37-49% RSS).

Not to be confused with DEFAULT_VM_MEMORY_OVERHEAD_MB which is a per-VM additive
constant for QEMU/gvproxy process memory."""

DEFAULT_CPU_OVERCOMMIT_RATIO: Final[float] = 2.0
"""Host-level CPU overcommit multiplier applied to the total VM pool budget.

budget = (host_cpus - reserve_cores) * overcommit_ratio

AI workloads average 11-17% CPU utilization, so 2x overcommit is safe while
keeping exec latency low.  The 2x default was confirmed by RBF surrogate
optimization (thin-plate spline, 16 samples) over a 4x4 latency-frontier sweep
(250 VMs per combo) on a 16-vCPU / 32GB EC2 instance with KVM + KSM
(scripts/benchmark_optimizer.py):

  Predicted optimal: CPU_OC=2.0, MEM_OC=4.6

Beyond 2x, setup and exec p95 degrade sharply under sustained load (250 VMs):
  CPU=2x: setup_p95=160ms, exec_p95=188ms, RSS=3.4%
  CPU=4x: setup_p95=719ms, exec_p95=336ms, RSS=5.9%
  CPU=8x: setup_p95=768ms, exec_p95=783ms, RSS=12.4%
  CPU=12x: setup_p95=31s, exec_p95=38s, RSS=36.9% (system collapse)

Not to be confused with DEFAULT_VM_CPU_OVERHEAD_CORES which is a per-VM additive
constant for QEMU/gvproxy process CPU."""

DEFAULT_HOST_MEMORY_RESERVE_RATIO: Final[float] = 0.08
"""Fraction of host memory reserved for OS and non-VM processes.
Scales with host size:
- 4GB host  → 320MB reserved  → 3.7GB for VMs
- 16GB host → 1.3GB reserved  → 14.7GB for VMs
- 64GB host → 5.1GB reserved  → 58.9GB for VMs

Python scheduler + OS overhead on a tuned server is ~300-800MB: the 1.3GB that
a 16GB host reserves is ample for both."""

DEFAULT_HOST_CPU_RESERVE_CORES: Final[float] = 0.5
"""CPU cores reserved for host processes (OS, Python scheduler, qemu-storage-daemon).
Fixed value, not a ratio — host infrastructure has roughly constant CPU consumption
regardless of host size. 0.5 cores covers typical observed usage with headroom."""

DEFAULT_VM_CPU_CORES: Final[int] = 1
"""CPU cores allocated per VM. Flows to three places:
- QEMU -smp (guest-visible vCPU count)
- cgroup cpu.max quota (host-side CPU time enforcement)
- Admission controller budget (capacity planning)"""

DEFAULT_DISK_MB: Final[int] = 1024
"""Default size of a VM's home disk (/home/user, ext4 on a raw sparse file).
Only blocks the guest writes take host space; usable space is ~95% of it."""

MIN_DISK_MB: Final[int] = 128
"""Smallest home disk: the ext4 journal, the REPL files and the files of a few
runs (packages live in the read-only L2 layer, not here)."""

DEFAULT_VM_MEMORY_OVERHEAD_MB: Final[int] = 80
"""QEMU + gvproxy process memory overhead added to guest memory for admission and cgroup limits.

Covers host-side process memory beyond guest RAM (QEMU VMM + gvproxy share a
per-VM cgroup). Uses ``microvm`` (x86) / ``virt`` (ARM) machine types — minimal
device set (virtio-mmio only), comparable to Firecracker (<5 MiB VMM overhead).

Measured breakdown:
  - QEMU binary + heap + glib event loop : ~20-30 MB
  - virtio-mmio ring buffers             : ~1 MB per device
  - qcow2 block layer (L2 + refcount)    : ~5-10 MB
  - Miscellaneous (serial, monitor, etc.) : ~2-5 MB
  - QEMU subtotal (minus guest RAM)       : ~35-50 MB
  - gvproxy (Go binary, network proxy)    : ~15-30 MB
  - Combined total                        : ~50-80 MB

80 MB matches the observed combined peak (50-80 MB): tight accounting admits
more concurrent VMs within the same memory budget.
If OOM-kills are observed, bump this — but first verify with:
  ``ps -o rss= -p <qemu_pid>`` and ``ps -o rss= -p <gvproxy_pid>``."""

DEFAULT_VM_CPU_OVERHEAD_CORES: Final[float] = 0.25
"""QEMU + gvproxy CPU overhead added to guest CPU for admission and cgroup limits.

Measured breakdown:
  - QEMU main loop + event processing      : ~2-5% of a core
  - IOThread (block I/O completion)         : ~1-3% of a core
  - gvproxy (TLS termination, DNS filter)   : ~5-15% of a core under network load
  - Haltpoll idle overhead (KVM)            : ~5-10% of a core
  - Combined total                          : ~13-33% of a core

0.25 provides ~60% headroom above observed peak (matching memory margin philosophy).
The gvproxy share applies only when gvproxy runs in the cgroup (network on); without
it the overhead is ~5-10% of a core, but every VM gets 0.25 (one value is simpler)."""

DEFAULT_TCG_TB_CACHE_SIZE_MB: Final[int] = 128
"""TCG translation block cache size in MB (tb-size in qemu_cmd.py). The cap, the most
QEMU can use, adds to the cgroup limit and the admission budget of each VM under TCG.

QEMU uses the cache pages only as it fills them, so the cap is a worst case, not a
cost: the RSS was the same with 32, 128 and 1024 MB caps. A full cache flushes every
translation (no per-block eviction), and the code is translated again. Measured use
(QEMU info jit): the kernel boot about 29 MB; a Python or JavaScript VM after three
execs 37-44 MB, with 0 flushes. Forced smaller caps at boot: 16 MB 1 flush (+20 ms),
8 MB 6 (+135 ms), 4 MB 23 (+360 ms). 128 MB is about three times that use, and an eighth
of QEMU's 1 GiB default in the budget."""

RESOURCE_ADMISSION_TIMEOUT_SECONDS: Final[float] = 120.0
"""Timeout for resource admission acquire (seconds). Blocks waiting for
resources to become available, then raises TransientError (CAPACITY)."""

GATE3_SELF_WAKE_INTERVAL_SECONDS: Final[float] = 10.0
"""Interval for Gate 3 self-wake timer (seconds). Periodically notifies
blocked waiters so they re-evaluate _blocking_gate with a fresh memory probe.
Closes the wakeup gap when external processes free memory but no local
release() triggers notify_all()."""

SUDO_RM_TIMEOUT_SECONDS: Final = 2.0
"""Bound of a `sudo -n rm -rf` of a VM working folder (permission_utils.sudo_rm at each
qemu-vm VM's destroy, private_run_dir._remove_folder in the sweep). Measured on EC2: 6 ms for
sudo itself, 138 ms for a folder with a 4 GiB file, 2 s for 300k files. A sudo stuck longer (a
hostname, PAM or SSSD lookup) is killed and the folder stays for the next sweep: a destroy,
close or start sweep never hangs on it."""

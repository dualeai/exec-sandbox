"""Constants for exec-sandbox configuration and limits."""

import enum
from dataclasses import dataclass, replace
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
    # Four attempts give a transient overlay daemon error (connection refused at startup)
    # time to clear
    max_attempts=4,
    # The guest agent ready check. Sized for CI nested virtualization (AMD nested KVM:
    # 50-90% CPU degradation per Cloud Hypervisor issue #4827). Bare-metal boots complete
    # in <1s; the timeout is for the full boot + guest-agent ready sequence under
    # worst-case contention
    boot_timeout_seconds=45,
    # Under CI CPU contention the host scheduler needs time to reschedule the QEMU
    # process: 20ms retries are too tight
    retry_min_seconds=0.1,
    # A 2 s cap with jitter lets the overlay daemon recover from transient
    # connection-refused errors (a daemon restart takes ~100-500ms)
    retry_max_seconds=2.0,
)
"""Default retry profile for user-facing VMs (scheduler)."""

RETRY_BACKGROUND: Final[RetryProfile] = replace(RETRY_USER_FACING, max_attempts=2, boot_timeout_seconds=30)
"""Retry profile for background VMs (warm pool, snapshot creation, L1 saves): the same
backoff, fewer attempts and a shorter boot timeout: no user request waits on these boots."""

GUEST_NETWORK_SUBNET: Final = "192.168.127.0/24"
"""The guest network of gvproxy-wrapper (passed as -subnet). The guest agent sets the guest
side at its boot (guest-agent/src/init.rs: GUEST_IP/24, its default route to the gateway):
change both together. A saved VM keeps the network of its boot."""
GUEST_GATEWAY_IP: Final = "192.168.127.1"
"""Gateway and DNS server of the guest network (gvproxy-wrapper -gateway-ip)."""
GUEST_GATEWAY_MAC: Final = "5a:94:ef:e4:0c:dd"
"""MAC address of the gateway (gvproxy-wrapper -gateway-mac)."""
GUEST_IP: Final = "192.168.127.2"
"""The guest's address: the target of each port forward."""
GUEST_MTU: Final = 1500
"""MTU of the guest network (gvproxy-wrapper -mtu)."""

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
caller sees OutcomeUnknownError (after dispatch) or TransientError
GUEST_UNAVAILABLE (before dispatch) — never a wrong result. That rare case is accepted
instead of delaying hung-guest detection by 8s on every execution. The
package-install path, which is always a first op on a network VM, adds its
own gate allowance (see disk_snapshot_manager).
This is a hard infrastructure backstop; it does not change the timeout of the
user's code (exit_code -1 at timeout_seconds)."""

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
Also used host-side to compute hard_timeout (+ EXECUTION_TIMEOUT_MARGIN_SECONDS + 8s
first-op network gate allowance, see disk_snapshot_manager)."""

PACKAGE_INSTALL_MAX_RETRIES: Final[int] = 3
"""Attempts of one package install, the first included (transient network errors)."""

L3_UPLOAD_STOP_GRACE_SECONDS: Final[float] = 30.0
"""How long DiskSnapshotManager.stop() (Scheduler exit) waits for the L3 uploads in flight
before it cancels them. An upload saves another host the build (a package install of
minutes); its compression and PUT of a few hundred MB take seconds. The bound keeps a store
that does not answer (botocore retries: minutes) from holding the exit; the abort of a
cancelled multipart upload adds at most s3_parts.ABORT_SECONDS."""

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
"""Maximum size of the code of one exec(): the guest counts bytes; the host check
(Session.exec, guest_agent_protocol) counts characters."""

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
"""Maximum relative file path length (POSIX PATH_MAX): the guest counts bytes; the
host check (guest_agent_protocol, QemuVM.write_file) counts characters.

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

With a boot log (on_boot_log: kernel loglevel=7), the kernel can emit 300-600
lines before init starts.
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
"""Max concurrent replenish boots per language, as a ratio of the pool size, rounded down
(at least 1): 2 concurrent boots for pool_size=5."""

WARM_POOL_TENANT_ID: Final[str] = "warm-pool"
"""Placeholder tenant ID for warm pool VMs."""

L1_SAVE_TENANT_ID: Final[str] = "l1-cache"
"""Tenant ID for sacrificial VMs created during L1 background saves."""

SCHEDULER_TENANT_ID: Final[str] = "exec-sandbox"
"""Tenant ID for user-facing VMs created by the scheduler."""

WARM_POOL_HEALTH_CHECK_INTERVAL: Final[int] = 15
"""Health check interval for warm VMs in seconds."""

WARM_POOL_HEALTH_CHECK_MAX_RETRIES: Final[int] = 3
"""Attempts of one health check, the first ping included, before the VM counts as unhealthy
(matches K8s failureThreshold)."""

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
"""Longest interval between query-migrate polls (5 ms): the polls start at 1 ms
and the interval doubles up to this value (migration_client)."""

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
QEMU pin in versions.lock (the QEMU that CI builds) is above this floor. Below the floor,
save and restore raise ConfigError MIGRATION_UNSUPPORTED (no legacy fallback)."""

MIGRATION_CAPABILITIES: Final[tuple[str, ...]] = ("multifd", "mapped-ram")
"""Migration capabilities that every save and every restore turn on: QEMU refuses a stream
whose sender set other ones ("received capability is off"). x-ignore-shared is set too,
with an explicit state (on for an L1 template only). Part of the L1 cache key: a change
here never restores an L1 entry saved with the old list."""

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

The 5.0x default comes from the 4x4 sweep of scripts/benchmark_optimizer.py (250 VMs
per combo, ranked by 10^6 / geomean of the admission, setup and exec p95) on a 16-vCPU /
32 GB KVM host (EC2, Xeon 6975P). At CPU=2x the memory ratio changed the results by 7%
or less, within the spread of one run:

  CPU=2x MEM=5x:   58.7 VMs/s, admit_p95=3.6s, setup_p95=282ms, exec_p95=132ms, RSS=4.8%
  CPU=2x MEM=1.5x: 58.8 VMs/s, admit_p95=3.6s, setup_p95=269ms, exec_p95=129ms, RSS=5.2%

README "Concurrent throughput" has the whole table.

Not to be confused with DEFAULT_VM_MEMORY_OVERHEAD_MB which is a per-VM additive
constant for QEMU/gvproxy process memory."""

DEFAULT_CPU_OVERCOMMIT_RATIO: Final[float] = 2.0
"""Host-level CPU overcommit multiplier applied to the total VM pool budget.

budget = (host_cpus - reserve_cores) * overcommit_ratio

AI workloads average 11-17% CPU utilization, so 2x overcommit is safe while
keeping exec latency low. The 2x default comes from the same 4x4 sweep as
DEFAULT_MEMORY_OVERCOMMIT_RATIO (scripts/benchmark_optimizer.py, 250 VMs per combo,
16-vCPU / 32 GB KVM host): CPU=2x gave the best results with every memory ratio.
Beyond 2x, throughput falls and setup and exec p95 grow (MEM=5x, then 1.5x for 12x):

  CPU=2x:  58.7 VMs/s, setup_p95=282ms, exec_p95=132ms, RSS=4.8%
  CPU=4x:  55.9 VMs/s, setup_p95=627ms, exec_p95=279ms, RSS=9.8%
  CPU=8x:  50.3 VMs/s, setup_p95=1.4s,  exec_p95=662ms, RSS=21.8%
  CPU=12x: 36.4 VMs/s, setup_p95=2.8s,  exec_p95=2.9s,  RSS=56.5%

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

0.25 lies inside the measured range: like DEFAULT_VM_MEMORY_OVERHEAD_MB, it is a
tight value, not the worst case. The gvproxy share applies only when gvproxy runs in the cgroup (network on); without
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

ARM_TCG_COUNTER_HZ: Final[int] = 62_500_000
"""Rate of the ARM generic timer of ARM64 TCG guests (qemu_cmd: cntfrq of the guest CPU): the
rate QEMU gives that model today, pinned. QEMU picks 62.5 MHz or 1 GHz by the model and the
machine-type version, and a load under another rate is accepted with no error (the guest
clock then runs 16 times too fast or too slow)."""

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

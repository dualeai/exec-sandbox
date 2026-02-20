"""Constants for exec-sandbox configuration and limits."""

from typing import Final

from exec_sandbox.models import Language

# ============================================================================
# VM Memory and Resource Defaults
# ============================================================================

DEFAULT_MEMORY_MB: Final[int] = 256
"""Default guest VM memory allocation in MB (reduced from 512MB for cost optimization)."""

MIN_MEMORY_MB: Final[int] = 128
"""Minimum guest VM memory in MB."""

TMPFS_SIZE_MB: Final[int] = 128
"""tmpfs /tmp size limit in MB (half of default VM memory)."""

# ============================================================================
# Execution Timeouts
# ============================================================================

DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
"""Default code execution timeout in seconds."""

MAX_TIMEOUT_SECONDS: Final[int] = 300
"""Maximum code execution timeout in seconds (5 minutes)."""

VM_BOOT_TIMEOUT_SECONDS: Final[int] = 30
"""VM boot timeout in seconds (guest agent ready check)."""

VM_BOOT_MAX_RETRIES: Final[int] = 3
"""Maximum retry attempts for VM boot failures (CPU contention resilience)."""

VM_BOOT_RETRY_MIN_SECONDS: Final[float] = 0.02
"""Minimum backoff between VM boot retries (20ms base)."""

VM_BOOT_RETRY_MAX_SECONDS: Final[float] = 0.5
"""Maximum backoff between VM boot retries (500ms cap with jitter)."""

GUEST_CONNECT_TIMEOUT_SECONDS: Final[int] = 5
"""Timeout for connecting to guest agent."""

EXECUTION_TIMEOUT_MARGIN_SECONDS: Final[int] = 8
"""Hard timeout margin above soft timeout (host watchdog protection).
Accounts for:
- Guest graceful termination grace period (5s SIGTERM→SIGKILL)
- JSON serialization, network transmission, clock skew (~2s)
- Safety buffer (~1s)
Must be >= guest-agent TERM_GRACE_PERIOD_SECONDS (5s) + overhead."""

PACKAGE_INSTALL_TIMEOUT_SECONDS: Final[int] = 120
"""Timeout for package installation in guest VM."""

# ============================================================================
# Code Execution Limits
# ============================================================================

MAX_CODE_SIZE: Final[int] = 1024 * 1024  # 1MB
"""Maximum size in bytes for code input."""

MAX_PACKAGES: Final[int] = 50
"""Maximum number of packages allowed per execution."""

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

MAX_STDOUT_SIZE: Final[int] = 1_000_000  # 1MB
"""Maximum stdout capture size in bytes."""

MAX_STDERR_SIZE: Final[int] = 100_000  # 100KB
"""Maximum stderr capture size in bytes."""

# ============================================================================
# File I/O
# ============================================================================

MAX_FILE_SIZE_BYTES: Final[int] = 500 * 1024 * 1024  # 500MB
"""Maximum file size for read/write operations.

With streaming zstd-compressed chunks, neither host nor guest buffers
the full file in memory (~128 KB peak).  The practical ceiling is guest
disk free space (rootfs_size + 100 MB), not RAM.
"""

MAX_FILE_PATH_LENGTH: Final[int] = 4096
"""Maximum relative file path length in bytes (POSIX PATH_MAX).

Individual filename components are limited to 255 bytes (NAME_MAX) by the
guest agent's validate_file_path(). This constant covers the full relative path.
"""

FILE_IO_TIMEOUT_SECONDS: Final[int] = 120
"""Timeout for file I/O operations in seconds."""


GUEST_SANDBOX_DIR: Final[str] = "/home/user"
"""Root directory for sandbox file operations in the guest VM."""

# ============================================================================
# Outbound Domain Filtering
# ============================================================================
# Note: dnsmasq approach was replaced by gvproxy OutboundAllow (DNS + TLS filtering)
# (see gvproxy.py start_gvproxy for rationale)

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

CONSOLE_LOG_MAX_BYTES: Final[int] = 8000
"""Maximum bytes to capture from VM console log for debugging (context/structured logs)."""

CONSOLE_LOG_PREVIEW_BYTES: Final[int] = 4000
"""Maximum bytes for console log preview in error messages."""

QEMU_OUTPUT_MAX_BYTES: Final[int] = 2000
"""Maximum bytes to capture from QEMU stdout/stderr."""

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

# ============================================================================
# Kernel Version Requirements
# ============================================================================

IO_URING_MIN_KERNEL_MAJOR: Final[int] = 5
"""Minimum Linux kernel major version for io_uring support."""

IO_URING_MIN_KERNEL_MINOR: Final[int] = 1
"""Minimum Linux kernel minor version for io_uring support (5.1+)."""

# ============================================================================
# Warm VM Pool
# ============================================================================

WARM_POOL_REPLENISH_CONCURRENCY_RATIO: Final[float] = 0.5
"""Max concurrent replenish boots as ratio of pool_size (50% = 2-3 concurrent for pool_size=5)."""

WARM_POOL_LANGUAGES: Final[tuple[Language, ...]] = (Language.PYTHON, Language.JAVASCRIPT)
"""Languages eligible for warm VM pool."""

WARM_POOL_TENANT_ID: Final[str] = "warm-pool"
"""Placeholder tenant ID for warm pool VMs."""

WARM_POOL_HEALTH_CHECK_INTERVAL: Final[int] = 10
"""Health check interval for warm VMs in seconds (matches K8s/Cloud Run periodSeconds default)."""

WARM_POOL_HEALTH_CHECK_MAX_RETRIES: Final[int] = 3
"""Maximum retry attempts before declaring VM unhealthy (matches K8s failureThreshold)."""

WARM_POOL_HEALTH_CHECK_RETRY_MIN_SECONDS: Final[float] = 0.1
"""Minimum backoff between health check retries."""

WARM_POOL_HEALTH_CHECK_RETRY_MAX_SECONDS: Final[float] = 2.0
"""Maximum backoff between health check retries."""

# ============================================================================
# Balloon Memory Management (for warm pool memory efficiency)
# ============================================================================

BALLOON_INFLATE_TARGET_MB: Final[int] = 96
"""Target guest memory in MB when inflating balloon for idle warm pool VMs.
Inflating the balloon reduces guest memory, allowing host to reclaim.

Note: 64MB was too aggressive - Alpine Linux idles at ~50MB, leaving only ~14MB
headroom for guest agent operations. Under memory pressure, the guest becomes
unresponsive and health checks timeout. 96MB provides ~46MB headroom while still
achieving 62% memory reduction (96MB vs 256MB default)."""

BALLOON_TOLERANCE_MB: Final[int] = 40
"""Tolerance in MB for balloon target polling. Allows early exit when balloon is
'close enough' to target, accounting for kernel overhead and slow CI runners."""

BALLOON_INFLATE_TIMEOUT_SECONDS: Final[float] = 5.0
"""Timeout for balloon inflate operation (reducing guest memory for idle pool)."""

BALLOON_DEFLATE_TIMEOUT_SECONDS: Final[float] = 5.0
"""Timeout for balloon deflate operation (restoring guest memory before execution)."""

# ============================================================================
# Overlay Pool
# ============================================================================

OVERLAY_POOL_FALLBACK_SIZE: Final[int] = 5
"""Overlay pool size when host resources are unknown (psutil unavailable).

Provides a reasonable default for pre-created overlays per base image
when the admission controller cannot determine host capacity."""

OVERLAY_POOL_SIZE_RATIO: Final[float] = 0.5
"""Overlay pool size as ratio of effective max VMs (50%).

With 10 max VMs, pool will have 5 pre-created overlays per base image.
Higher ratio = more pool hits, lower ratio = less disk usage.
"""

OVERLAY_POOL_REPLENISH_BATCH_SIZE: Final[int] = 8
"""Max concurrent overlay creations during startup and replenishment.

Benchmarked on NVMe SSD (30 overlays):
- 4: 0.50s (59.8/sec) - too conservative
- 8: 0.41s (73.2/sec) - 22% faster, good balance
- 16: 0.36s (82.9/sec) - 38% faster but aggressive
- 32: 0.42s (71.7/sec) - contention kicks in

8 balances throughput vs compatibility with slower storage.
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

PORT_FORWARD_MIN_HOST_PORT: Final[int] = 1024
"""Minimum host port for port forwarding (unprivileged ports only)."""

PORT_FORWARD_BIND_HOST: Final[str] = "127.0.0.1"
"""Host address to bind forwarded ports to (localhost only for security)."""

GVPROXY_API_TIMEOUT_SECONDS: Final[float] = 5.0
"""Timeout for gvproxy HTTP API requests (port forward expose/unexpose)."""

# ============================================================================
# File Transfer Streaming
# ============================================================================

FILE_TRANSFER_CHUNK_SIZE: Final[int] = 128 * 1024  # 128KB
"""Raw chunk size for streaming file transfers (before zstd compression).

128KB balances fewer frames (halves syscalls, JSON parses, base64 en/decodes
vs 64KB) while staying within virtio queue depth (128-256 descriptors).
On-wire size after base64+JSON is ~175KB per frame.
"""

FILE_TRANSFER_ZSTD_LEVEL: Final[int] = 3
"""Zstd compression level for file transfers (1=fast, 22=max; 3=good balance)."""

# ============================================================================
# Resource Admission & Overcommit
# ============================================================================

DEFAULT_MEMORY_OVERCOMMIT_RATIO: Final[float] = 1.5
"""Default memory overcommit ratio. Effective memory budget =
host_total * (1 - host_reserve_ratio) * overcommit_ratio.
AI agent workloads are bursty and mostly idle (avg memory 18-22%),
making overcommit safe."""

DEFAULT_CPU_OVERCOMMIT_RATIO: Final[float] = 4.0
"""Default CPU overcommit ratio. Effective CPU budget =
host_cpu_count * ratio. AI workloads average 11-17% CPU utilization."""

DEFAULT_HOST_MEMORY_RESERVE_RATIO: Final[float] = 0.1
"""Fraction of host memory reserved for OS and non-VM processes.
Scales with host size:
- 4GB host  → 400MB reserved  → 3.6GB for VMs
- 16GB host → 1.6GB reserved  → 14.4GB for VMs
- 64GB host → 6.4GB reserved  → 57.6GB for VMs"""

DEFAULT_VM_CPU_CORES: Final[int] = 1
"""CPU cores allocated per VM. Flows to three places:
- QEMU -smp (guest-visible vCPU count)
- cgroup cpu.max quota (host-side CPU time enforcement)
- Admission controller budget (capacity planning)"""

RESOURCE_MONITOR_INTERVAL_SECONDS: Final[float] = 5.0
"""Interval between resource monitor ticks (seconds)."""

RESOURCE_ADMISSION_TIMEOUT_SECONDS: Final[float] = 120.0
"""Timeout for resource admission acquire (seconds). Blocks waiting for
resources to become available, then raises VmCapacityError."""

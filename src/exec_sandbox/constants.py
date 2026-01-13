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

MAX_MEMORY_MB: Final[int] = 2048
"""Maximum guest VM memory in MB."""

TMPFS_SIZE_MB: Final[int] = 128
"""tmpfs /tmp size limit in MB (half of default VM memory)."""

CGROUP_MEMORY_OVERHEAD_MB: Final[int] = 200
"""QEMU process overhead added to guest memory for cgroup limits."""

TCG_TB_CACHE_SIZE_MB: Final[int] = 512
"""TCG translation block cache size in MB (must match tb-size in vm_manager.py).
QEMU 5.0+ defaults to 1GB, we use 512MB for better CI compatibility.
See: https://blueprints.launchpad.net/nova/+spec/control-qemu-tb-cache"""

CGROUP_PIDS_LIMIT: Final[int] = 100
"""Maximum PIDs in cgroup (fork bomb prevention)."""

# ============================================================================
# Execution Timeouts
# ============================================================================

DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
"""Default code execution timeout in seconds."""

MAX_TIMEOUT_SECONDS: Final[int] = 300
"""Maximum code execution timeout in seconds (5 minutes)."""

VM_BOOT_TIMEOUT_SECONDS: Final[int] = 30
"""VM boot timeout in seconds (guest agent ready check)."""

GUEST_CONNECT_TIMEOUT_SECONDS: Final[int] = 5
"""Timeout for connecting to guest agent (TCP)."""

EXECUTION_TIMEOUT_MARGIN_SECONDS: Final[int] = 8
"""Hard timeout margin above soft timeout (host watchdog protection).
Accounts for:
- Guest graceful termination grace period (5s SIGTERM→SIGKILL)
- JSON serialization, network transmission, clock skew (~2s)
- Safety buffer (~1s)
Must be >= guest-agent TERM_GRACE_PERIOD_SECONDS (5s) + overhead."""

PACKAGE_INSTALL_TIMEOUT_SECONDS: Final[int] = 120
"""Timeout for package installation in guest VM."""

ULIMIT_CPU_TIME_SECONDS: Final[int] = 60
"""CPU time limit for ulimit (fallback when cgroups unavailable)."""

ULIMIT_MEMORY_MULTIPLIER: Final[int] = 14
"""Virtual memory multiplier for ulimit (guest_mb * 14 for TCG overhead)."""

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
# Communication
# ============================================================================

TCP_GUEST_PORT: Final[int] = 5000
"""TCP port for guest agent communication."""

# ============================================================================
# DNS and Domain Filtering
# ============================================================================
# Note: dnsmasq approach was replaced by gvproxy DNS zones
# (see vm_manager.py _start_gvproxy for rationale)

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

ERRNO_READ_ONLY_FILESYSTEM: Final[int] = 30
"""errno code for read-only filesystem (OSError.errno)."""

ERRNO_PERMISSION_DENIED: Final[int] = 13
"""errno code for permission denied (OSError.errno)."""

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

WARM_POOL_SIZE_RATIO: Final[float] = 0.25
"""Warm pool size as ratio of max_concurrent_vms (25% = 2-3 VMs for default=10)."""

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
# Memory Optimization (Balloon)
# ============================================================================

BALLOON_DEFLATE_MIN_MB: Final[int] = 64
"""Minimum memory target during balloon deflation (64MB floor)."""

BALLOON_DEFLATE_TIMEOUT_SECONDS: Final[float] = 10.0
"""Timeout for balloon deflation before snapshot."""

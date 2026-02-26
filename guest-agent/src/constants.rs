//! Constants and statics for the guest agent.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicU64};

use tokio::sync::{Mutex as TokioMutex, Notify};

use crate::repl::ReplState;
use crate::types::Language;

pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");
pub(crate) const CMD_PORT_PATH: &str = "/dev/virtio-ports/org.dualeai.cmd";
pub(crate) const EVENT_PORT_PATH: &str = "/dev/virtio-ports/org.dualeai.event";

// Execution limits
pub(crate) const MAX_CODE_SIZE_BYTES: usize = 1_000_000; // 1MB max code size
pub(crate) const MAX_PACKAGE_OUTPUT_BYTES: usize = 50_000; // 50KB max package install output
pub(crate) const MAX_TIMEOUT_SECONDS: u64 = 300; // 5 minutes max execution timeout

// Sandbox user identity — all code execution runs as this non-root user.
// Matches the "user" account created in the VM image (see build-qcow2.sh).
pub(crate) const SANDBOX_UID: u32 = 1000;
pub(crate) const SANDBOX_GID: u32 = 1000;

// Connection limits
pub(crate) const MAX_REQUEST_SIZE_BYTES: usize = 16_000_000; // 16MB max single request JSON
pub(crate) const RETRY_DELAY_MS: u64 = 50; // 50ms retry delay on transient errors
// Bounded channel size for the guest→host write queue.
pub(crate) const WRITE_QUEUE_SIZE: usize = 128;
pub(crate) const READ_TIMEOUT_MS: u64 = 18000; // 18s > 15s health check interval

// Host disconnection backoff configuration
// INITIAL: 1ms for fast reconnect (virtio-serial port typically ready in 1-2ms).
// MAX: 1000ms to keep orphan VM CPU low (~1% vs 5x higher at 200ms).
// Exponential backoff reaches 1000ms after ~10 attempts (~2s).
pub(crate) const INITIAL_BACKOFF_MS: u64 = 1;
pub(crate) const MAX_BACKOFF_MS: u64 = 1000;

// Environment variable limits
pub(crate) const MAX_ENV_VARS: usize = 100;
pub(crate) const MAX_ENV_VAR_NAME_LENGTH: usize = 256;
pub(crate) const MAX_ENV_VAR_VALUE_LENGTH: usize = 4096;

// Package limits
pub(crate) const MAX_PACKAGES: usize = 50;
pub(crate) const MAX_PACKAGE_NAME_LENGTH: usize = 214; // PyPI limit
pub(crate) const PACKAGE_INSTALL_TIMEOUT_SECONDS: u64 = 300;

// Streaming configuration
pub(crate) const FLUSH_INTERVAL_MS: u64 = 50;
pub(crate) const DRAIN_TIMEOUT_MS: u64 = 5;
pub(crate) const MAX_BUFFER_SIZE_BYTES: usize = 64 * 1024; // 64KB

// File I/O limits
pub(crate) const MAX_FILE_SIZE_BYTES: usize = 500 * 1024 * 1024; // 500 MiB
pub(crate) const MAX_FILE_PATH_LENGTH: usize = 4096; // POSIX PATH_MAX
pub(crate) const MAX_FILE_NAME_BYTES: usize = 255; // POSIX NAME_MAX
pub(crate) const SANDBOX_ROOT: &str = "/home/user";
pub(crate) const PYTHON_SITE_PACKAGES: &str = "/usr/lib/python3/site-packages";
pub(crate) const NODE_MODULES_SYSTEM: &str = "/usr/local/lib/node_modules";

// Runtime paths — shared between spawn.rs (REPL commands).
pub(crate) const JEMALLOC_LIB: &str = "/usr/lib/libjemalloc.so.2";
pub(crate) const PYTHON_HOME: &str = "/opt/python";
pub(crate) const BUN_BIN_PATH: &str = "/usr/local/bin/bun";
pub(crate) const BASH_BIN_PATH: &str = "/bin/bash";

// File transfer streaming
pub(crate) const FILE_TRANSFER_CHUNK_SIZE: usize = 128 * 1024;
pub(crate) const FILE_TRANSFER_ZSTD_LEVEL: i32 = 3;

// Graceful termination configuration
pub(crate) const TERM_GRACE_PERIOD_SECONDS: u64 = 5;

/// E4: Quiet mode — suppress non-essential log_info! on the boot critical path.
/// Each eprintln! triggers an MMIO trap through the console device (~0.1-5ms per line
/// depending on console type: virtio-console, PL011, ISA serial).
/// Enabled by `init.quiet=1` on the kernel command line.
pub(crate) static QUIET_MODE: LazyLock<bool> = LazyLock::new(|| {
    std::fs::read_to_string("/proc/cmdline")
        .unwrap_or_default()
        .split_ascii_whitespace()
        .any(|p| p == "init.quiet=1")
});

/// Blocked dangerous environment variables (case-insensitive check at validation time).
pub(crate) static BLOCKED_ENV_VARS: &[&str] = &[
    // Dynamic linker (arbitrary code execution)
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "LD_BIND_NOW",
    "LD_DEBUG",
    "LD_DEBUG_OUTPUT",
    "LD_USE_LOAD_BIAS",
    "LD_PROFILE",
    "LD_ORIGIN_PATH",
    "LD_AOUT_LIBRARY_PATH",
    "LD_AOUT_PRELOAD",
    // glibc tunables - CVE-2023-4911
    "GLIBC_TUNABLES",
    // Node.js runtime
    "NODE_OPTIONS",
    "NODE_REPL_HISTORY",
    // Python runtime
    "PYTHONPATH",
    "PYTHONINSPECT",
    "PYTHONWARNINGS",
    "PYTHONSTARTUP",
    "PYTHONHOME",
    // Shell environment execution
    "BASH_ENV",
    "ENV",
    "PROMPT_COMMAND",
    "IFS",
    "CDPATH",
    "PS4",
    // Path manipulation
    "PATH",
    // glibc/system hooks
    "GCONV_PATH",
    "HOSTALIASES",
    "LOCPATH",
    "NLSPATH",
    "RESOLV_HOST_CONF",
    "RES_OPTIONS",
    "TMPDIR",
    "TZDIR",
    "MALLOC_CHECK_",
    "MALLOC_TRACE",
    "MALLOC_PERTURB_",
];

/// Network readiness gate: signals when ip + gvproxy setup is complete.
/// ExecuteCode/InstallPackages wait on this; Ping/file I/O respond immediately.
pub(crate) static NETWORK_READY: AtomicBool = AtomicBool::new(false);
pub(crate) static NETWORK_NOTIFY: LazyLock<Notify> = LazyLock::new(Notify::new);

/// Monotonic counter for sentinel IDs. Combined with nanosecond timestamp
/// to produce unique, unpredictable IDs without the uuid crate dependency.
pub(crate) static SENTINEL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Module-level REPL storage. Persists across guest agent reconnect cycles (READ_TIMEOUT_MS=18s).
pub(crate) static REPL_STATES: LazyLock<TokioMutex<HashMap<Language, ReplState>>> =
    LazyLock::new(|| TokioMutex::new(HashMap::new()));

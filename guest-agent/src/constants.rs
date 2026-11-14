//! Constants and statics for the guest agent.

use std::sync::LazyLock;

pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");
pub(crate) const CMD_PORT_PATH: &str = "/dev/virtio-ports/org.dualeai.cmd";
pub(crate) const EVENT_PORT_PATH: &str = "/dev/virtio-ports/org.dualeai.event";

// Execution limits
pub(crate) const MAX_CODE_SIZE_BYTES: usize = 1024 * 1024; // 1 MiB max code size
pub(crate) const MAX_PACKAGE_OUTPUT_BYTES: usize = 50_000; // 50KB max package install output
pub(crate) const MAX_TIMEOUT_SECONDS: u64 = 300; // 5 minutes max execution timeout

/// The deadline of an exec or a package install of `requested` seconds: 0
/// means the default, MAX_TIMEOUT_SECONDS (the host accepts 0 for both).
pub(crate) fn effective_timeout_secs(requested: u64) -> u64 {
    if requested > 0 {
        requested
    } else {
        MAX_TIMEOUT_SECONDS
    }
}

// Sandbox user identity — all code execution runs as this non-root user.
// Matches the "user" account created in the VM image (see build-qcow2.sh).
pub(crate) const SANDBOX_UID: u32 = 1000;
pub(crate) const SANDBOX_GID: u32 = 1000;

// Connection limits
pub(crate) const MAX_REQUEST_SIZE_BYTES: usize = 16_000_000; // 16MB max single request JSON
// Pause before reopening the ports after a connection error.
pub(crate) const RETRY_DELAY_MS: u64 = 50;
// Bounded channel size for the guest→host write queue.
pub(crate) const WRITE_QUEUE_SIZE: usize = 128;

// Backoff between attempts to open the virtio-serial ports (boot, or after a
// port error). A host close does not reopen them (see CommandPort).
// INITIAL: 1ms for a fast open (virtio-serial port typically ready in 1-2ms).
// MAX: 1000ms to keep CPU low while a port stays broken.
// Exponential backoff reaches 1000ms after 10 attempts (~1s of waits).
pub(crate) const INITIAL_BACKOFF_MS: u64 = 1;
pub(crate) const MAX_BACKOFF_MS: u64 = 1000;

// Environment variable limits
pub(crate) const MAX_ENV_VARS: usize = 100;
pub(crate) const MAX_ENV_VAR_NAME_LENGTH: usize = 256;
pub(crate) const MAX_ENV_VAR_VALUE_LENGTH: usize = 4096;

// Package limits
pub(crate) const MAX_PACKAGES: usize = 50;
// npm name limit, checked on the full spec (name + version).
pub(crate) const MAX_PACKAGE_NAME_LENGTH: usize = 214;

// Streaming configuration. Exec output is sent every FLUSH_INTERVAL_MS, and at
// once when a buffer reaches MAX_BUFFER_SIZE_BYTES or when no output was sent
// in the last FLUSH_INTERVAL_MS (execute.rs).
// DRAIN_TIMEOUT_MS bounds each read of the drain after a timeout SIGTERM.
pub(crate) const FLUSH_INTERVAL_MS: u64 = 50;
pub(crate) const DRAIN_TIMEOUT_MS: u64 = 5;
pub(crate) const MAX_BUFFER_SIZE_BYTES: usize = 64 * 1024; // 64KB

// Output size limits (guest-enforced). Exceeding either raises CmdError::output_limit
// to the host (InvalidInputError OUTPUT_LIMIT in Python). The REPL is preserved for
// session reuse.
pub(crate) const MAX_STDOUT_BYTES: usize = 1_000_000; // 1 MB
pub(crate) const MAX_STDERR_BYTES: usize = 100_000; // 100 KB

// File I/O limits
pub(crate) const MAX_FILE_SIZE_BYTES: usize = 500 * 1024 * 1024; // 500 MiB
pub(crate) const MAX_FILE_PATH_LENGTH: usize = 4096; // POSIX PATH_MAX
pub(crate) const MAX_FILE_NAME_BYTES: usize = 255; // POSIX NAME_MAX
pub(crate) const SANDBOX_ROOT: &str = "/home/user";
pub(crate) const PYTHON_SITE_PACKAGES: &str = "/usr/lib/python3/site-packages";
pub(crate) const NODE_MODULES_SYSTEM: &str = "/usr/local/lib/node_modules";

// Runtime paths of the REPL interpreters (spawn.rs); packages.rs also uses PYTHON_HOME.
pub(crate) const JEMALLOC_LIB: &str = "/usr/lib/libjemalloc.so.2";
pub(crate) const PYTHON_HOME: &str = "/opt/python";
pub(crate) const BUN_BIN_PATH: &str = "/usr/local/bin/bun";
pub(crate) const BASH_BIN_PATH: &str = "/bin/bash";

// File transfer streaming
pub(crate) const FILE_TRANSFER_CHUNK_SIZE: usize = 128 * 1024;
pub(crate) const FILE_TRANSFER_ZSTD_LEVEL: i32 = 3;

// Graceful termination configuration
pub(crate) const TERM_GRACE_PERIOD_SECONDS: u64 = 5;

// After an installer (uv, bun) ends and its group is killed, its output pipes close at
// once; a process that left its group can hold them. The readers wait at least this long
// after the end, also when the install deadline is closer.
pub(crate) const INSTALLER_OUTPUT_GRACE_MS: u64 = 1000;

// Longest wait for a process after SIGKILL (wait_child_bounded). A process in
// uninterruptible sleep dies only when its I/O ends. Fits in the host margin
// (EXECUTION_TIMEOUT_MARGIN_SECONDS = 10) with TERM_GRACE_PERIOD_SECONDS.
pub(crate) const KILL_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

// Longest wait for the writer task of a pipelined file write: for a free slot
// for each chunk, at file_end for the end of the write, and for a task that
// ended early. A writer stuck on the disk would else stop the agent, which reads
// one command at a time. Below the host's ack wait (FILE_IO_TIMEOUT_SECONDS =
// 120), so a write that failed gets an error reply, not a deadline. A write
// whose rename stalls after its commit claim gets no reply: the host's deadline
// then gives an unknown outcome, the truth.
pub(crate) const FILE_WRITE_STALL_TIMEOUT_SECONDS: u64 = 60;

/// Quiet mode — suppress non-essential log_info! on the boot critical path.
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
///
/// SECURITY NOTE: These names are visible via `strings(1)` on the compiled binary.
/// This is an accepted risk because:
///   1. The list contains well-known dangerous vars documented in every Linux
///      hardening guide — no information advantage is gained by reading the binary.
///   2. The rootfs is read-only EROFS; the binary cannot be patched in-guest.
///   3. The VM itself is the primary isolation boundary; this blocklist is
///      defense-in-depth, not the sole security control.
///   4. Obfuscation (e.g. compile-time hashing) would add complexity and a
///      normalization step (case-insensitive matching) for negligible benefit.
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

/// Longest network setup at boot (init::NETWORK_STATE is set within it).
pub(crate) const NETWORK_SETUP_TIMEOUT_SECONDS: u64 = 7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_timeout_is_the_default_deadline() {
        // One rule for an install and an exec: 0 is the default deadline
        assert_eq!(effective_timeout_secs(0), 300);
        assert_eq!(effective_timeout_secs(1), 1);
        assert_eq!(effective_timeout_secs(120), 120);
    }
}

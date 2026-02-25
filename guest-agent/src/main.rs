//! QEMU Guest Agent
//!
//! Lightweight async agent running inside QEMU microVMs.
//! Communicates with host via virtio-serial for:
//! - Package installation (pip, bun)
//! - Code execution via persistent REPL (Python, JavaScript, Shell)
//! - Health checks
//!
//! All code execution uses persistent REPL wrappers (not per-exec processes).
//! Each language has a long-lived interpreter process that receives code via
//! a length-prefixed stdin protocol and signals completion via unique
//! sentinels on stderr (nanosecond timestamp + counter). State (variables,
//! imports, functions) persists across executions within the same REPL instance.
//!
//! Uses tokio for fully async, non-blocking I/O.
//! Communication via dual virtio-serial ports:
//! - /dev/virtio-ports/org.dualeai.cmd (host → guest, read-only)
//! - /dev/virtio-ports/org.dualeai.event (guest → host, write-only)

mod constants;

/// Logging macros — severity-leveled wrappers around eprintln!.
///
/// Output format: `[LEVEL] message` — consistent with tiny-init's convention
/// for greppable serial console output across the full boot chain.
///
/// `log_error!` and `log_warn!` always print (rare, important).
/// `log_info!` is gated on `init.quiet=1` to avoid MMIO/UART trap overhead
/// on the boot critical path.
macro_rules! log_error {
    ($($arg:tt)*) => { eprintln!("[ERROR] {}", format_args!($($arg)*)) };
}
macro_rules! log_warn {
    ($($arg:tt)*) => { eprintln!("[WARN] {}", format_args!($($arg)*)) };
}
macro_rules! log_info {
    ($($arg:tt)*) => {
        if !*crate::constants::QUIET_MODE {
            eprintln!("[INFO] {}", format_args!($($arg)*));
        }
    };
}

mod connection;
mod error;
mod file_io;
mod init;
mod packages;
mod repl;
mod types;
mod validation;

use constants::*;

pub(crate) fn monotonic_ms() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1000 + (ts.tv_nsec as u64) / 1_000_000
}

/// Listen on virtio-serial ports with reconnection loop.
async fn listen_virtio_serial() -> Result<(), Box<dyn std::error::Error>> {
    use tokio::fs::OpenOptions;

    log_info!(
        "Guest agent opening virtio-serial ports: cmd={CMD_PORT_PATH}, event={EVENT_PORT_PATH}"
    );

    let mut backoff_ms = INITIAL_BACKOFF_MS;

    loop {
        let cmd_file = match connection::NonBlockingFile::open_read(CMD_PORT_PATH) {
            Ok(f) => {
                log_info!("Guest agent connected to command port (read, non-blocking)");
                f
            }
            Err(e) => {
                log_error!("Failed to open command port: {e}, retrying in {backoff_ms}ms...");
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
        };

        match cmd_file.is_host_connected() {
            Ok(true) => {
                log_info!("Host is connected, proceeding with connection setup");
                backoff_ms = INITIAL_BACKOFF_MS;
            }
            Ok(false) => {
                log_warn!("Host not connected (POLLHUP), waiting {backoff_ms}ms before retry...");
                drop(cmd_file);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
            Err(e) => {
                log_error!(
                    "Failed to check host connection status: {e}, retrying in {backoff_ms}ms..."
                );
                drop(cmd_file);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
        }

        let event_file = match OpenOptions::new().write(true).open(EVENT_PORT_PATH).await {
            Ok(f) => {
                log_info!("Guest agent connected to event port (write)");
                f
            }
            Err(e) => {
                log_error!("Failed to open event port: {e}, retrying in {backoff_ms}ms...");
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
        };

        log_info!("[timing] agent_connected: {}ms (monotonic)", monotonic_ms());

        let mut cmd_file = cmd_file;
        if let Err(e) = connection::handle_connection(&mut cmd_file, event_file).await {
            log_error!("Connection error: {e}, reopening ports...");
            tokio::time::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS)).await;
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t_start = monotonic_ms();
    log_info!("[timing] agent_start: {}ms (monotonic)", t_start);

    // Phase 1 — minimal setup (env vars only, <1ms)
    init::setup_phase1();
    let t_phase1 = monotonic_ms();
    log_info!(
        "[timing] agent_phase1: {}ms ({}ms elapsed)",
        t_phase1,
        t_phase1 - t_start
    );

    // Phase 2 core runs on spawn_blocking while we concurrently attempt
    // the first virtio port open. Virtio ports appear ~5-15ms into boot
    // while init does filesystem/sysctl work — overlapping saves 2-5ms.
    // Phase 2 must complete before any request processing (Ping).
    let phase2_handle = tokio::task::spawn_blocking(|| {
        init::setup_phase2_core();
        let t_phase2 = monotonic_ms();
        log_info!("[timing] agent_phase2_core: {}ms", t_phase2);
    });

    // Background tasks — none block Ping/file I/O readiness.
    // ExecuteCode/InstallPackages gate on NETWORK_READY only.
    tokio::spawn(init::setup_network_background());
    tokio::spawn(init::setup_zram_background());
    tokio::spawn(init::reap_zombies());

    log_info!("Guest agent starting (dual ports: cmd={CMD_PORT_PATH}, event={EVENT_PORT_PATH})...");

    // Wait for phase2 to finish before entering the connection loop.
    // This ensures all mounts, sysctls, and dev setup are complete before Ping.
    phase2_handle.await.ok();

    listen_virtio_serial().await
}

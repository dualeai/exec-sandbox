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

mod connection;
mod constants;
mod error;
mod file_io;
mod init;
mod packages;
mod repl;
mod types;
mod validation;

use constants::*;

/// Listen on virtio-serial ports with reconnection loop.
async fn listen_virtio_serial() -> Result<(), Box<dyn std::error::Error>> {
    use tokio::fs::OpenOptions;

    eprintln!(
        "Guest agent opening virtio-serial ports: cmd={CMD_PORT_PATH}, event={EVENT_PORT_PATH}"
    );

    let mut backoff_ms = INITIAL_BACKOFF_MS;

    loop {
        let cmd_file = match connection::NonBlockingFile::open_read(CMD_PORT_PATH) {
            Ok(f) => {
                eprintln!("Guest agent connected to command port (read, non-blocking)");
                f
            }
            Err(e) => {
                eprintln!("Failed to open command port: {e}, retrying in {backoff_ms}ms...");
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
        };

        match cmd_file.is_host_connected() {
            Ok(true) => {
                eprintln!("Host is connected, proceeding with connection setup");
                backoff_ms = INITIAL_BACKOFF_MS;
            }
            Ok(false) => {
                eprintln!("Host not connected (POLLHUP), waiting {backoff_ms}ms before retry...");
                drop(cmd_file);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
            Err(e) => {
                eprintln!(
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
                eprintln!("Guest agent connected to event port (write)");
                f
            }
            Err(e) => {
                eprintln!("Failed to open event port: {e}, retrying in {backoff_ms}ms...");
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
        };

        let mut cmd_file = cmd_file;
        if let Err(e) = connection::handle_connection(&mut cmd_file, event_file).await {
            eprintln!("Connection error: {e}, reopening ports...");
            tokio::time::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS)).await;
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::process::id() == 1 {
        eprintln!("Guest agent running as PID 1 (init)...");
        init::setup_init_environment();
        eprintln!("Enabling zombie reaper...");
        tokio::spawn(init::reap_zombies());
    }

    eprintln!("Guest agent starting (dual ports: cmd={CMD_PORT_PATH}, event={EVENT_PORT_PATH})...");
    listen_virtio_serial().await
}

//! QEMU Guest Agent
//!
//! Lightweight async agent running inside QEMU microVMs.
//! Communicates with host via virtio-serial for:
//! - Package installation (uv pip for Python, bun for JavaScript)
//! - Code execution via persistent REPL (Python, JavaScript, Shell)
//! - File transfer (write, read, list under /home/user)
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

mod clock;
mod connection;
mod error;
mod file_io;
mod init;
mod oom_guard;
mod packages;
mod repl;
mod types;
mod validation;

use constants::*;

/// CLOCK_MONOTONIC in milliseconds (time since boot), for `[timing]` log lines.
pub(crate) fn monotonic_ms() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1000 + (ts.tv_nsec as u64) / 1_000_000
}

/// Force the kernel CRNG to reseed from its entropy input pool.
///
/// After L1 snapshot restore, the kernel CRNG state is byte-for-byte identical
/// across all VMs restored from the same snapshot (CRNG_RESEED_INTERVAL is 300s,
/// so the hwrng kthread won't feed new entropy in time). This ioctl forces an
/// immediate reseed so each VM diverges before any user code calls getrandom().
///
/// Called at two points for defense-in-depth:
/// 1. `listen_virtio_serial()` — each time the ports open (covers cold boot)
/// 2. `handle_connection()` — before each command dispatch but a file chunk
///    (covers L1 restore, where the VM resumes mid-loop with the ports already open)
///
/// Cost: ~11μs per call, negligible vs connection/exec overhead.
fn reseed_crng() {
    // RNDRESEEDCRNG = _IO('R', 0x07) = 0x5207
    // libc::ioctl request type differs by target (c_ulong on glibc, c_int on musl)
    const RNDRESEEDCRNG: libc::c_int = 0x5207;

    unsafe {
        let fd = libc::open(c"/dev/urandom".as_ptr(), libc::O_WRONLY);
        if fd < 0 {
            log_warn!("reseed_crng: failed to open /dev/urandom");
            return;
        }
        let ret = libc::ioctl(fd, RNDRESEEDCRNG as _);
        libc::close(fd);
        if ret < 0 {
            log_warn!(
                "reseed_crng: ioctl RNDRESEEDCRNG failed (errno={})",
                *libc::__errno_location()
            );
        } else {
            log_info!("reseed_crng: kernel CRNG reseeded");
        }
    }
}

/// Serve the host over the virtio-serial ports.
///
/// The ports stay open while they work. A host that closes its sockets (VM
/// teardown, an L1 restore, a channel reset) ends one connection; the agent
/// then serves the next host on the same ports. Closing a port would make the
/// guest kernel drop the bytes that the host sends before QEMU sees the close
/// (see connection::CommandPort). Only a port error makes the agent reopen
/// the ports.
async fn listen_virtio_serial() -> Result<(), Box<dyn std::error::Error>> {
    log_info!(
        "Guest agent opening virtio-serial ports: cmd={CMD_PORT_PATH}, event={EVENT_PORT_PATH}"
    );

    let mut backoff_ms = INITIAL_BACKOFF_MS;

    loop {
        let (mut cmd_port, event_port) = match open_ports().await {
            Ok(ports) => ports,
            Err(e) => {
                log_error!(
                    "Failed to open virtio-serial ports: {e}, retrying in {backoff_ms}ms..."
                );
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
        };
        backoff_ms = INITIAL_BACKOFF_MS;
        reseed_crng();
        log_info!("[timing] agent_connected: {}ms (monotonic)", monotonic_ms());

        loop {
            // Each connection writes through its own descriptor of the event
            // port; `event_port` keeps the port open between connections.
            let served = match event_port.try_clone().await {
                Ok(event_file) => connection::handle_connection(&mut cmd_port, event_file).await,
                Err(e) => Err(e.into()),
            };
            if let Err(e) = served {
                log_error!("Connection error: {e}, reopening ports...");
                tokio::time::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS)).await;
                break;
            }
        }
    }
}

/// Open the command port (read) and the event port (write).
///
/// No host needs to be connected: bytes that a host sends before the agent
/// reads stay in QEMU or in the guest kernel until the agent reads them.
async fn open_ports() -> std::io::Result<(connection::CommandPort, tokio::fs::File)> {
    let cmd_port = connection::CommandPort::open(CMD_PORT_PATH)?;
    log_info!("Guest agent opened command port (read, non-blocking)");
    let event_port = tokio::fs::OpenOptions::new()
        .write(true)
        .open(EVENT_PORT_PATH)
        .await?;
    log_info!("Guest agent opened event port (write)");
    Ok((cmd_port, event_port))
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

    // Phase 2 core performs blocking filesystem and sysctl setup off the async
    // runtime. It must complete before any request processing (Ping).
    let phase2_handle = tokio::task::spawn_blocking(|| {
        init::setup_phase2_core();
        let t_phase2 = monotonic_ms();
        log_info!("[timing] agent_phase2_core: {}ms", t_phase2);
    });

    log_info!("Guest agent starting (dual ports: cmd={CMD_PORT_PATH}, event={EVENT_PORT_PATH})...");

    // Wait for phase2 to finish before entering the connection loop.
    // This ensures all mounts, sysctls, and dev setup are complete before Ping.
    phase2_handle.await?;

    // Start independent work only after phase 2 has prepared /dev. Zram setup
    // is awaited: it removes its setup-only node and seals /dev read-only
    // before the agent opens the long-lived virtio port descriptors, so the
    // ports open through the final /dev mount.
    tokio::spawn(init::reap_zombies());
    tokio::spawn(init::setup_network_background());
    init::setup_zram_and_seal_dev().await;

    listen_virtio_serial().await
}

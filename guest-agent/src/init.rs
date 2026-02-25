//! Init-time setup: PID 1 environment, mounts, network, zombie reaping.

use std::process::Command as StdCommand;

use std::sync::atomic::Ordering;

use crate::constants::{NETWORK_NOTIFY, NETWORK_READY, SANDBOX_GID, SANDBOX_UID};

// ============================================================================
// Zombie reaping
// ============================================================================

/// Reap zombie processes when running as PID 1.
pub(crate) async fn reap_zombies() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigchld = match signal(SignalKind::child()) {
        Ok(s) => s,
        Err(e) => {
            log_warn!("Failed to register SIGCHLD handler: {e}");
            return;
        }
    };

    loop {
        sigchld.recv().await;
        loop {
            let pid = unsafe { libc::waitpid(-1, std::ptr::null_mut(), libc::WNOHANG) };
            match pid {
                p if p > 0 => continue,
                _ => break,
            }
        }
    }
}

// ============================================================================
// Init environment setup
// ============================================================================

/// A1: Phase 1 — minimal setup for Ping response (fast, <1ms).
/// Sets environment variables only. Called before listen_virtio_serial().
pub(crate) fn setup_phase1() {
    // SAFETY: called at startup before any threads are spawned
    unsafe { std::env::set_var("PATH", "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin") };
    unsafe { std::env::set_var("UV_NO_CACHE", "1") };
}

/// Phase 2 core — sync init: chmod, mounts, critical sysctl, dev setup (~10ms).
/// Network, zram, and deferred sysctl run in background to unblock Ping.
pub(crate) fn setup_phase2_core() {
    // CIS Benchmark 6.1.x compliance (A2: libc::chmod, no fork/exec)
    chmod_paths(0o755, &["/etc", "/usr", "/var", "/sbin", "/bin"]);
    chmod_paths(
        0o644,
        &[
            "/etc/passwd",
            "/etc/group",
            "/etc/resolv.conf",
            "/etc/hosts",
        ],
    );
    chmod_paths(0o640, &["/etc/shadow"]);

    mount_home_tmpfs();
    // ALL /proc/sys writes MUST happen before mount_readonly_paths() which
    // makes /proc/sys read-only. This includes sysctl hardening AND zram
    // VM tuning (swappiness, page-cluster, etc.).
    apply_sysctl_critical();
    apply_sysctl_deferred();
    apply_zram_vm_tuning();
    // Match EROFS 16KB physical cluster size (-C16384 in mkfs.erofs).
    //
    // EROFS decompresses in pcluster-sized units, so readahead must match:
    //   - Below pcluster: wastes a decompression cycle (kernel fetches less than
    //     one pcluster, forcing a second I/O to complete decompression)
    //   - Above pcluster: prefetches unneeded clusters, wasting bandwidth/memory
    //   - Equal: exactly one decompression per readahead — optimal
    //
    // 16KB was chosen over 64KB for better random-read performance during REPL
    // sessions (Python imports hundreds of scattered 4-16KB .pyc files):
    //   - 16KB clusters: 123 MiB/s random reads (best in EROFS benchmarks)
    //   - 64KB clusters:  67 MiB/s random reads (~46% slower)
    //   - Sequential reads: ~850 MiB/s at 16KB vs 907 at 64KB (~6% tradeoff)
    // Ref: erofs-utils PERFORMANCE.md (Debian rootfs dataset)
    //
    // Set on all vd* block devices — device naming varies by architecture
    // (ARM MMIO enumerates in reverse order), so we don't assume vda=base.
    for entry in std::fs::read_dir("/sys/block")
        .into_iter()
        .flatten()
        .flatten()
    {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("vd") {
            let path = entry.path().join("queue/read_ahead_kb");
            let _ = std::fs::write(path, "16");
        }
    }
    // /dev writes (symlinks, shm mount) must happen before mount_readonly_paths()
    // which now makes /dev read-only.
    setup_dev_symlinks();
    setup_dev_shm();
    mount_readonly_paths();
}

/// Background zram setup: device wait, compression, mkswap, swapon, VM tuning.
/// E3: Moved off critical path — zram is only needed under memory pressure,
/// not for Ping/file I/O readiness.
pub(crate) async fn setup_zram_background() {
    tokio::task::spawn_blocking(setup_zram_swap).await.ok();
}

/// Background network setup: ip config + gvproxy verification.
/// Runs on spawn_blocking (uses StdCommand for ip, I/O-bound).
/// Marks NETWORK_READY when complete; ExecuteCode/InstallPackages gate on this.
pub(crate) async fn setup_network_background() {
    tokio::task::spawn_blocking(|| {
        setup_network();
    })
    .await
    .ok();
    mark_network_ready();
}

fn mark_network_ready() {
    NETWORK_READY.store(true, Ordering::Release);
    NETWORK_NOTIFY.notify_waiters();
}

/// Wait until network setup is complete. No-op if already ready.
pub(crate) async fn wait_for_network() {
    if NETWORK_READY.load(Ordering::Acquire) {
        return;
    }
    // Re-check after registering the notified future to avoid TOCTOU race
    let notified = NETWORK_NOTIFY.notified();
    if NETWORK_READY.load(Ordering::Acquire) {
        return;
    }
    notified.await;
}

/// Mount tmpfs on /home/user — writable scratch space on read-only rootfs.
fn mount_home_tmpfs() {
    let ret = unsafe {
        let source = std::ffi::CString::new("tmpfs").unwrap();
        let target = std::ffi::CString::new("/home/user").unwrap();
        let fstype = std::ffi::CString::new("tmpfs").unwrap();
        let data = std::ffi::CString::new(format!(
            "mode=0755,uid={SANDBOX_UID},gid={SANDBOX_GID},noswap"
        ))
        .unwrap();
        libc::mount(
            source.as_ptr(),
            target.as_ptr(),
            fstype.as_ptr(),
            libc::MS_NOSUID | libc::MS_NODEV,
            data.as_ptr() as *const libc::c_void,
        )
    };
    if ret != 0 {
        log_warn!(
            "tmpfs mount on /home/user failed: {}",
            std::io::Error::last_os_error()
        );
    }
}

/// Read-only bind remounts for CIS hardening.
fn mount_readonly_paths() {
    let rw_mode = std::fs::read_to_string("/proc/cmdline")
        .unwrap_or_default()
        .split_ascii_whitespace()
        .any(|p| p == "init.rw=1");

    if rw_mode {
        log_info!("init.rw=1: skipping /usr, /bin, /sbin read-only remounts");
    } else {
        for path in [c"/usr", c"/bin", c"/sbin"] {
            let _ = mount_readonly(path);
        }
    }

    // A5: Removed per-mount log_info! calls — each is an expensive serial write
    // /dev is safe to make read-only here: setup_dev_symlinks() and setup_dev_shm()
    // run before this function. listen_virtio_serial() opens existing device files
    // (not creating new ones), so read-only /dev doesn't block it. The background
    // zram setup opens /dev/zram0 (an existing block device) for write — the block
    // device layer bypasses VFS permission checks, so RO bind mount doesn't block it.
    // The remove_file("/dev/zram0") at cleanup is already `let _ =` (ignores EROFS).
    for path in [
        c"/dev",
        c"/etc/hosts",
        c"/etc/resolv.conf",
        c"/proc/sys",
        c"/proc/sysrq-trigger",
    ] {
        let _ = mount_readonly(path);
    }
}

/// Configure loopback and eth0 network interfaces.
/// No wait loop for eth0 — configure if present, skip if not.
/// Called from spawn_blocking in setup_network_background().
fn setup_network() {
    let _ = StdCommand::new("ip")
        .args(["link", "set", "lo", "up"])
        .status();

    if std::path::Path::new("/sys/class/net/eth0").exists() {
        let _ = StdCommand::new("ip")
            .args(["link", "set", "eth0", "up"])
            .status();
        let _ = StdCommand::new("ip")
            .args(["addr", "add", "192.168.127.2/24", "dev", "eth0"])
            .status();
        let _ = StdCommand::new("ip")
            .args(["route", "add", "default", "via", "192.168.127.1"])
            .status();

        // Verify gvproxy connectivity. ExecuteCode/InstallPackages gate on
        // NETWORK_READY, so network is guaranteed ready before code/package ops.
        verify_gvproxy();
    } else {
        log_warn!("eth0 not found, network unavailable");
    }
}

// ============================================================================
// Deferred operations (B1, B4, B5 — moved off tiny-init critical path)
// ============================================================================

/// B5: Create /dev/fd and /dev/std* symlinks.
/// Required for bash process substitution <() and /dev/std* portability.
fn setup_dev_symlinks() {
    use std::os::unix::fs::symlink;
    let _ = symlink("/proc/self/fd", "/dev/fd");
    let _ = symlink("/proc/self/fd/0", "/dev/stdin");
    let _ = symlink("/proc/self/fd/1", "/dev/stdout");
    let _ = symlink("/proc/self/fd/2", "/dev/stderr");
}

/// B4: Mount /dev/shm for POSIX shared memory / semaphores.
/// Only needed for Python multiprocessing and similar use cases.
fn setup_dev_shm() {
    let ret = unsafe {
        let source = std::ffi::CString::new("tmpfs").unwrap();
        let target = std::ffi::CString::new("/dev/shm").unwrap();
        let fstype = std::ffi::CString::new("tmpfs").unwrap();
        let data = std::ffi::CString::new("size=64M,noswap").unwrap();
        libc::mount(
            source.as_ptr(),
            target.as_ptr(),
            fstype.as_ptr(),
            libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC,
            data.as_ptr() as *const libc::c_void,
        )
    };
    if ret != 0 {
        // Non-fatal: only needed for multiprocessing
        let _ = std::fs::create_dir_all("/dev/shm");
        let ret2 = unsafe {
            let source = std::ffi::CString::new("tmpfs").unwrap();
            let target = std::ffi::CString::new("/dev/shm").unwrap();
            let fstype = std::ffi::CString::new("tmpfs").unwrap();
            let data = std::ffi::CString::new("size=64M,noswap").unwrap();
            libc::mount(
                source.as_ptr(),
                target.as_ptr(),
                fstype.as_ptr(),
                libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC,
                data.as_ptr() as *const libc::c_void,
            )
        };
        if ret2 != 0 {
            log_warn!("/dev/shm mount failed: {}", std::io::Error::last_os_error());
        }
    }
}

/// Lazily write REPL wrapper scripts on first spawn (deferred from init for boot speed).
/// Scripts are compile-time constants; std::sync::Once ensures write-once semantics.
pub(crate) fn ensure_repl_wrappers() {
    use crate::constants::SANDBOX_ROOT;
    use crate::repl::wrappers::*;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::fs::write(format!("{SANDBOX_ROOT}/_repl.py"), PYTHON_REPL_WRAPPER);
        let _ = std::fs::write(format!("{SANDBOX_ROOT}/_repl.mjs"), JS_REPL_WRAPPER);
        let _ = std::fs::write(format!("{SANDBOX_ROOT}/_repl.sh"), SHELL_REPL_WRAPPER);
    });
}

/// B1: Full zram device setup (moved from tiny-init).
/// Includes: device wait, compression config, disksize, mkswap, swapon, VM tuning.
fn setup_zram_swap() {
    use std::io::Write;
    use std::path::Path;
    use std::thread;
    use std::time::Duration;

    // Syscall numbers for swapon(2)
    #[cfg(target_arch = "x86_64")]
    const SYS_SWAPON: libc::c_long = 167;
    #[cfg(target_arch = "aarch64")]
    const SYS_SWAPON: libc::c_long = 224;
    const SWAP_FLAG_PREFER: libc::c_int = 0x8000;

    // Wait for zram device to appear after module load (loaded by tiny-init)
    // Exponential backoff: 0.5+1+2+4+8+16 = 31.5ms max
    let delays_us = [500, 1000, 2000, 4000, 8000, 16000];
    let found = {
        let mut ok = false;
        for &delay_us in &delays_us {
            if Path::new("/sys/block/zram0").exists() {
                ok = true;
                break;
            }
            thread::sleep(Duration::from_micros(delay_us));
        }
        if !ok {
            ok = Path::new("/sys/block/zram0").exists();
        }
        ok
    };
    if !found {
        log_warn!("[zram] device not found, skipping");
        return;
    }

    // Compression algorithm fallback chain: lz4 -> lzo-rle -> lzo
    let algo = ["lz4", "lzo-rle", "lzo"]
        .iter()
        .find(|a| std::fs::write("/sys/block/zram0/comp_algorithm", a).is_ok());
    if algo.is_none() {
        log_warn!("[zram] failed to set compression algorithm, skipping");
        return;
    }

    // Kernel 6.16+: algorithm-specific tuning via sysfs
    let _ = std::fs::write("/sys/block/zram0/algorithm_params", "level=1");

    let mem_kb: u64 = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0);

    if mem_kb == 0 {
        log_warn!("[zram] failed to read MemTotal, skipping");
        return;
    }

    // disksize = 50% of RAM (in bytes)
    let zram_size = mem_kb * 512;
    if std::fs::write("/sys/block/zram0/disksize", zram_size.to_string()).is_err() {
        log_warn!("[zram] failed to set disksize, skipping");
        return;
    }

    // Build and write swap header (mkswap equivalent)
    let header = match build_swap_header(zram_size) {
        Some(h) => h,
        None => {
            log_warn!("[zram] device too small for swap, skipping");
            return;
        }
    };
    let header_result = (|| -> std::io::Result<()> {
        let mut f = std::fs::OpenOptions::new().write(true).open("/dev/zram0")?;
        f.write_all(&header)
    })();
    if let Err(e) = header_result {
        log_warn!("[zram] mkswap failed: {e}, skipping");
        return;
    }

    // swapon with high priority
    let dev = std::ffi::CString::new("/dev/zram0").unwrap();
    let ret = unsafe { libc::syscall(SYS_SWAPON, dev.as_ptr(), SWAP_FLAG_PREFER | 100) };
    if ret < 0 {
        log_warn!(
            "[zram] swapon failed (errno={}), skipping",
            std::io::Error::last_os_error()
        );
        return;
    }

    // Security: remove device node after swapon
    let _ = std::fs::remove_file("/dev/zram0");

    // VM tuning for zram is pre-applied in apply_zram_vm_tuning() before
    // /proc/sys is made read-only. No sysctl writes needed here.
}

/// Build a swap header for the given device size.
/// Returns a 4096-byte header or None if the size is too small.
fn build_swap_header(device_size: u64) -> Option<Vec<u8>> {
    let pages = (device_size / 4096).saturating_sub(1) as u32;
    if pages < 10 {
        return None; // kernel rejects tiny swap
    }
    let mut header = vec![0u8; 4096];
    // SWAPSPACE2 signature at offset 4086
    header[4086..4096].copy_from_slice(b"SWAPSPACE2");
    // version = 1
    header[1024..1028].copy_from_slice(&1u32.to_le_bytes());
    // last_page (0-indexed: total_pages - 1, since page 0 is header)
    header[1028..1032].copy_from_slice(&pages.to_le_bytes());
    Some(header)
}

// ============================================================================
// Sysctl hardening (moved from tiny-init for boot latency; modules_disabled stays in tiny-init)
// ============================================================================

/// E3: Critical sysctl — must be applied before any request handling.
/// These block privilege escalation and info-leak vectors that sandbox
/// code could exploit immediately.
const SYSCTL_CRITICAL: &[(&str, &str)] = &[
    // eBPF: CVE-2020-8835, CVE-2021-3490, CVE-2021-31440, CVE-2023-2163
    ("/proc/sys/kernel/unprivileged_bpf_disabled", "2"),
    // User namespaces: CVE-2022-0185, CVE-2023-0386
    ("/proc/sys/kernel/unprivileged_userns_clone", "0"),
    ("/proc/sys/user/max_user_namespaces", "0"),
    // Kernel address exposure
    ("/proc/sys/kernel/kptr_restrict", "2"),
    // dmesg restriction
    ("/proc/sys/kernel/dmesg_restrict", "1"),
    // Perf events
    ("/proc/sys/kernel/perf_event_paranoid", "3"),
    // BPF JIT hardening
    ("/proc/sys/net/core/bpf_jit_harden", "2"),
    // userfaultfd
    ("/proc/sys/vm/unprivileged_userfaultfd", "0"),
    // YAMA ptrace_scope
    ("/proc/sys/kernel/yama/ptrace_scope", "2"),
];

/// E3: Deferred sysctl — defense-in-depth settings safe to apply after Ping.
/// These harden filesystem and limit attack surface but aren't exploitable
/// in the window between Ping-ready and background completion.
const SYSCTL_DEFERRED: &[(&str, &str)] = &[
    // Filesystem link protections
    ("/proc/sys/fs/protected_symlinks", "1"),
    ("/proc/sys/fs/protected_hardlinks", "1"),
    ("/proc/sys/fs/protected_fifos", "2"),
    ("/proc/sys/fs/protected_regular", "2"),
    // suid_dumpable
    ("/proc/sys/fs/suid_dumpable", "0"),
    // SysRq keyboard combos
    ("/proc/sys/kernel/sysrq", "0"),
    // Thread bomb mitigation
    ("/proc/sys/kernel/threads-max", "1200"),
];

fn apply_sysctl_critical() {
    for &(path, value) in SYSCTL_CRITICAL {
        let _ = std::fs::write(path, value);
    }
}

fn apply_sysctl_deferred() {
    for &(path, value) in SYSCTL_DEFERRED {
        let _ = std::fs::write(path, value);
    }
}

/// Pre-apply zram VM tuning sysctls before /proc/sys is made read-only.
/// These values are safe to set even before zram device setup completes:
/// the kernel uses defaults until swapon, then these take effect.
fn apply_zram_vm_tuning() {
    let mem_kb: u64 = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0);

    if mem_kb == 0 {
        return;
    }

    let _ = std::fs::write("/proc/sys/vm/page-cluster", "0");
    let _ = std::fs::write("/proc/sys/vm/swappiness", "180");
    let _ = std::fs::write(
        "/proc/sys/vm/min_free_kbytes",
        (mem_kb * 4 / 100).to_string(),
    );
    let _ = std::fs::write("/proc/sys/vm/overcommit_memory", "0");
    let _ = std::fs::write("/proc/sys/vm/defrag_mode", "1");
}

// ============================================================================
// Helpers
// ============================================================================

/// Verify gvproxy connectivity with exponential backoff.
/// Blocks phase 2 completion, so ExecuteCode/InstallPackages wait for network.
///
/// DNS-first verification (no fork/exec):
/// UDP DNS query to gateway:53 — any response proves both L3 routing AND DNS proxy
/// are ready (can't get a UDP response without working L3). We query `_probe.internal`
/// which is guaranteed to hit gvproxy's DNS proxy without external dependencies.
/// Any response (including NXDOMAIN) = ready.
///
/// If DNS fails after all retries, a single TCP probe to gateway:1 distinguishes
/// "no L3" from "L3 ok but DNS broken" — for diagnostics only.
fn verify_gvproxy() {
    use std::net::{SocketAddr, TcpStream, UdpSocket};

    // DNS probe: proves both L3 routing and DNS proxy readiness in one shot.
    // Minimal DNS query for "_probe.internal" (type A, class IN).
    #[rustfmt::skip]
    let dns_query: &[u8] = &[
        0x00, 0x01, // Transaction ID
        0x01, 0x00, // Flags: standard query, recursion desired
        0x00, 0x01, // Questions: 1
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Answers/Authority/Additional: 0
        // QNAME: _probe.internal
        6, b'_', b'p', b'r', b'o', b'b', b'e',
        8, b'i', b'n', b't', b'e', b'r', b'n', b'a', b'l',
        0, // root label
        0x00, 0x01, // QTYPE: A
        0x00, 0x01, // QCLASS: IN
    ];
    let delays_ms: &[u64] = &[1, 2, 5, 10, 25, 50, 100, 200, 500, 1000];
    for (i, &delay_ms) in delays_ms.iter().enumerate() {
        if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
            sock.set_read_timeout(Some(std::time::Duration::from_millis(200)))
                .ok();
            if sock.send_to(dns_query, "192.168.127.1:53").is_ok() {
                let mut buf = [0u8; 512];
                if sock.recv(&mut buf).is_ok() {
                    log_info!("Network verified via DNS (attempt {})", i + 1);
                    return;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
    }

    // DNS failed — single TCP probe for diagnostics (distinguish L3 vs DNS failure)
    let addr: SocketAddr = "192.168.127.1:1".parse().unwrap();
    let l3_ok = match TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(200)) {
        Ok(_) => true,
        Err(e) => e.raw_os_error() == Some(libc::ECONNREFUSED),
    };
    if l3_ok {
        log_warn!("gvproxy L3 reachable but DNS not responding, external connectivity may fail");
    } else {
        log_warn!("gvproxy unreachable (no L3), network will not work");
    }
}

/// A2: Set file mode on multiple paths using libc::chmod (no fork/exec overhead).
fn chmod_paths(mode: libc::mode_t, paths: &[&str]) {
    for path in paths {
        if let Ok(cpath) = std::ffi::CString::new(*path) {
            unsafe { libc::chmod(cpath.as_ptr(), mode) };
        }
    }
}

/// Bind-mount a path onto itself and remount read-only with nosuid.
fn mount_readonly(path: &std::ffi::CStr) -> bool {
    unsafe {
        let ret = libc::mount(
            path.as_ptr(),
            path.as_ptr(),
            std::ptr::null(),
            libc::MS_BIND | libc::MS_REC,
            std::ptr::null(),
        );
        if ret != 0 {
            log_warn!(
                "bind mount {} failed: {}",
                path.to_string_lossy(),
                std::io::Error::last_os_error()
            );
            return false;
        }
        let ret = libc::mount(
            std::ptr::null(),
            path.as_ptr(),
            std::ptr::null(),
            libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY | libc::MS_NOSUID,
            std::ptr::null(),
        );
        if ret != 0 {
            log_warn!(
                "RO remount {} failed: {}",
                path.to_string_lossy(),
                std::io::Error::last_os_error()
            );
            return false;
        }
        true
    }
}

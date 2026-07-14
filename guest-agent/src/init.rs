//! Init-time setup: PID 1 environment, mounts, network, zombie reaping.

use std::process::Command as StdCommand;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::constants::{
    NETWORK_FAILED, NETWORK_NOTIFY, NETWORK_PENDING, NETWORK_READY, NETWORK_SETUP_TIMEOUT_SECONDS,
    NETWORK_STATE, SANDBOX_GID, SANDBOX_UID, ZRAM_FAILED, ZRAM_READY, ZRAM_STATE,
};

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

/// Phase 2 core — sync init: chmod, mounts, sysctl hardening, dev setup (~10ms).
/// Called from spawn_blocking before port is opened. Network and zram run in background.
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
    apply_sysctl_non_critical();
    apply_zram_vm_tuning();
    // Readahead tuning for virtio-blk + EROFS.
    //
    // EROFS pcluster is 16KB (-C16384). Readahead set to 16KB (1 pcluster)
    // to minimize per-file readahead buffer memory. Tested 128KB vs 16KB:
    // no measurable wall-time difference (bottleneck is Mach kernel hv_trap
    // overhead, not I/O batch size). Lower readahead = less kernel page cache
    // pressure from speculative reads that may never be consumed.
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

/// Zram setup + /dev sealing, run to completion before the virtio listener
/// opens its long-lived ports (main.rs awaits this). No guest-side deadline:
/// the blocking worker cannot be cancelled, and abandoning it would leave an
/// armed pressure guard and live swap inside a VM already published as failed.
/// A wedged worker is the host boot watchdog's failure to own — it discards
/// the VM. There is no no-swap path.
pub(crate) async fn setup_zram_and_seal_dev() {
    let zram_active = match tokio::task::spawn_blocking(setup_zram_swap).await {
        Ok(ZramSetupOutcome::Active) => true,
        Ok(ZramSetupOutcome::Failed) => {
            log_warn!("[zram] setup failed; marking VM unavailable");
            false
        }
        Err(error) => {
            log_warn!("[zram] setup worker failed: {error}");
            false
        }
    };
    let dev_read_only = mount_readonly(c"/dev");
    if !dev_read_only {
        log_warn!("[zram] failed to seal /dev read-only; marking VM unavailable");
    }
    let state = if zram_active && dev_read_only {
        ZRAM_READY
    } else {
        ZRAM_FAILED
    };
    ZRAM_STATE.store(state, Ordering::Release);
}

/// Background network setup: ip config + gvproxy verification.
/// Runs on spawn_blocking (uses StdCommand for ip, I/O-bound). The blocking
/// worker cannot be cancelled, so publish a terminal failure if it exceeds the
/// deadline; the host will discard the VM instead of leaving a request waiting.
pub(crate) async fn setup_network_background() {
    let t0 = crate::monotonic_ms();
    let state = match tokio::time::timeout(
        Duration::from_secs(NETWORK_SETUP_TIMEOUT_SECONDS),
        tokio::task::spawn_blocking(setup_network),
    )
    .await
    {
        Ok(Ok(Ok(()))) => NETWORK_READY,
        Ok(Ok(Err(error))) => {
            log_warn!("network setup failed: {error}");
            NETWORK_FAILED
        }
        Ok(Err(error)) => {
            log_warn!("network setup worker failed: {error}");
            NETWORK_FAILED
        }
        Err(_) => {
            log_warn!(
                "network setup exceeded {}s; marking VM unavailable",
                NETWORK_SETUP_TIMEOUT_SECONDS
            );
            NETWORK_FAILED
        }
    };
    publish_network_state(state);
    let t_done = crate::monotonic_ms();
    if state == NETWORK_READY {
        log_info!(
            "[timing] network_ready: {}ms ({}ms elapsed)",
            t_done,
            t_done - t0
        );
    } else {
        log_warn!(
            "[timing] network_failed: {}ms ({}ms elapsed)",
            t_done,
            t_done - t0
        );
    }
}

fn publish_network_state(state: u8) {
    NETWORK_STATE.store(state, Ordering::Release);
    NETWORK_NOTIFY.notify_waiters();
}

fn current_network_state() -> Result<bool, &'static str> {
    decode_network_state(NETWORK_STATE.load(Ordering::Acquire))
}

fn decode_network_state(state: u8) -> Result<bool, &'static str> {
    match state {
        NETWORK_READY => Ok(true),
        NETWORK_FAILED => Err("network setup failed"),
        NETWORK_PENDING => Ok(false),
        _ => Err("network setup entered an invalid state"),
    }
}

/// Wait until network setup reaches a terminal state. Intentionally offline VMs
/// are ready after loopback setup; a failed/stuck worker is an infrastructure
/// error so the host can destroy and replace the VM.
pub(crate) async fn wait_for_network() -> Result<(), &'static str> {
    if current_network_state()? {
        return Ok(());
    }

    // Re-check after registering the notified future to avoid TOCTOU race
    let notified = NETWORK_NOTIFY.notified();
    if current_network_state()? {
        return Ok(());
    }

    tokio::time::timeout(
        Duration::from_secs(NETWORK_SETUP_TIMEOUT_SECONDS + 1),
        notified,
    )
    .await
    .map_err(|_| "network setup readiness timed out")?;

    if current_network_state()? {
        Ok(())
    } else {
        Err("network setup notification arrived without a terminal state")
    }
}

fn decode_zram_state(state: u8) -> Result<(), &'static str> {
    match state {
        ZRAM_READY => Ok(()),
        ZRAM_FAILED => Err("zram safety setup failed"),
        _ => Err("zram safety state is not terminal"),
    }
}

/// Capped-zram setup publishes a terminal state before the virtio listener
/// starts (main.rs awaits it), so commands check it synchronously and fail
/// closed on anything but active swap. In particular, `swapon` cannot overlap
/// untrusted process execution before monitor startup.
pub(crate) fn require_memory_safety() -> Result<(), &'static str> {
    decode_zram_state(ZRAM_STATE.load(Ordering::Acquire))
}

/// Process-spawning commands require both boot workers' outcomes. Zram is
/// terminal before the listener starts; only network setup is still concurrent.
pub(crate) async fn wait_for_untrusted_readiness() -> Result<(), &'static str> {
    require_memory_safety()?;
    wait_for_network().await
}

/// Mount tmpfs on /home/user — writable scratch space on read-only rootfs.
///
/// Sized to 40% of RAM with per-UID quota enforcement (`usrquota` +
/// `usrquota_block_hardlimit`) to prevent sparse file inflation attacks.
/// Quota must be set at initial mount time — cannot be added via remount.
///
/// 40% (vs 50% prior): most sandbox code uses <30MB; 40% of 192MB = 77MB
/// which provides ample headroom while reducing tmpfs metadata overhead.
fn mount_home_tmpfs() {
    // Page-align (round down) so the mount size is an exact multiple of PAGE_SIZE.
    let page_mask = !(page_size() - 1);
    // 40% of RAM: mem_kb * 1024 * 2 / 5 = mem_kb * 2048 / 5
    let tmpfs_bytes = (read_mem_total_kb() * 2048 / 5) & page_mask;
    let ret = unsafe {
        let source = std::ffi::CString::new("tmpfs").unwrap();
        let target = std::ffi::CString::new("/home/user").unwrap();
        let fstype = std::ffi::CString::new("tmpfs").unwrap();
        let data = std::ffi::CString::new(format!(
            "mode=0755,uid={SANDBOX_UID},gid={SANDBOX_GID},size={tmpfs_bytes},usrquota,usrquota_block_hardlimit={tmpfs_bytes},noswap"
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
    // /dev is sealed by setup_zram_and_seal_dev() only after zram has removed its
    // setup-only block-device node. Untrusted execution remains gated until that
    // read-only remount succeeds.
    for path in [
        c"/etc/hosts",
        c"/etc/resolv.conf",
        c"/proc/sys",
        c"/proc/sysrq-trigger",
    ] {
        let _ = mount_readonly(path);
    }

    // Mask sensitive /proc files (bind-mount /dev/null → reads return empty).
    // Safe to mask here: rw_mode and QUIET_MODE already read /proc/cmdline above.
    for path in [
        // Boot params: rootfstype, init flags, console config (reconnaissance)
        c"/proc/cmdline",
        // Exact kernel version string (aids CVE matching against guest kernel)
        c"/proc/version",
        // Interrupt counters per CPU — thermal side-channel attack vector
        // (Docker masked this in Mar 2025, GHSA-6fw5-f8r9-fgfm)
        c"/proc/interrupts",
        // Kernel keyring — NOT namespaced; mask even though currently empty
        c"/proc/keys",
        // High-resolution timer internals — timing side-channel
        // (already EPERM via dmesg_restrict, mask as defense-in-depth)
        c"/proc/timer_list",
    ] {
        let _ = mount_mask(path);
    }
}

/// Configure loopback and eth0 network interfaces.
/// No wait loop for eth0 — configure if present, skip if not.
/// Called from spawn_blocking in setup_network_background().
fn setup_network() -> Result<(), String> {
    run_ip("loopback_up", &["link", "set", "lo", "up"])?;

    if std::path::Path::new("/sys/class/net/eth0").exists() {
        run_ip("eth0_up", &["link", "set", "eth0", "up"])?;
        run_ip(
            "eth0_address",
            &["addr", "add", "192.168.127.2/24", "dev", "eth0"],
        )?;
        run_ip(
            "default_route",
            &["route", "add", "default", "via", "192.168.127.1"],
        )?;

        // Verify gvproxy connectivity. ExecuteCode/InstallPackages gate on the
        // untrusted-readiness check (network + zram), so a failure here makes
        // every code/package op fail rather than run without networking.
        verify_gvproxy().map_err(str::to_string)?;
    } else {
        log_info!("eth0 not found, intentionally offline network is ready");
    }
    Ok(())
}

fn run_ip(stage: &str, args: &[&str]) -> Result<(), String> {
    let started = crate::monotonic_ms();
    log_info!("[network] stage={stage} action=start at={started}ms");
    match StdCommand::new("ip").args(args).status() {
        Ok(status) if status.success() => {
            log_info!(
                "[network] stage={stage} action=complete elapsed={}ms",
                crate::monotonic_ms() - started
            );
            Ok(())
        }
        Ok(status) => Err(format!(
            "stage={stage} status={status} elapsed={}ms",
            crate::monotonic_ms() - started
        )),
        Err(error) => Err(format!(
            "stage={stage} spawn_failed={error} elapsed={}ms",
            crate::monotonic_ms() - started
        )),
    }
}

// ============================================================================
// Deferred operations (B4, B5 — moved off tiny-init critical path)
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
///
/// Sized to 40% of RAM with per-UID quota enforcement (`usrquota` +
/// `usrquota_block_hardlimit`) to prevent sparse file inflation attacks.
/// Quota must be set at initial mount time — cannot be added via remount.
fn setup_dev_shm() {
    // Page-align (round down) so the mount size is an exact multiple of PAGE_SIZE.
    let page_mask = !(page_size() - 1);
    // 40% of RAM (matching /home/user sizing)
    let shm_bytes = (read_mem_total_kb() * 2048 / 5) & page_mask;
    let opts = format!("size={shm_bytes},usrquota,usrquota_block_hardlimit={shm_bytes},noswap");

    fn try_mount_shm(opts: &str) -> libc::c_int {
        unsafe {
            let source = std::ffi::CString::new("tmpfs").unwrap();
            let target = std::ffi::CString::new("/dev/shm").unwrap();
            let fstype = std::ffi::CString::new("tmpfs").unwrap();
            let data = std::ffi::CString::new(opts).unwrap();
            libc::mount(
                source.as_ptr(),
                target.as_ptr(),
                fstype.as_ptr(),
                libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC,
                data.as_ptr() as *const libc::c_void,
            )
        }
    }

    if try_mount_shm(&opts) != 0 {
        // Non-fatal: only needed for multiprocessing. Retry after mkdir.
        let _ = std::fs::create_dir_all("/dev/shm");
        if try_mount_shm(&opts) != 0 {
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

/// Terminal outcome of `setup_zram_swap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ZramSetupOutcome {
    Active,
    Failed,
}

const ZRAM_DISK_PERCENT: u64 = 40;
const ZRAM_MEM_PERCENT: u64 = 20;

fn zram_geometry(mem_kb: u64) -> (u64, u64) {
    // Guests are hundreds of MB; u64 arithmetic on MemTotal cannot overflow.
    let mem_bytes = mem_kb * 1024;
    (
        mem_bytes * ZRAM_DISK_PERCENT / 100,
        mem_bytes * ZRAM_MEM_PERCENT / 100,
    )
}

fn setup_zram_swap() -> ZramSetupOutcome {
    use std::io::Write;
    use std::path::Path;
    use std::thread;
    use std::time::Duration;

    // Syscall numbers for swapon(2)
    #[cfg(target_arch = "x86_64")]
    const SYS_SWAPON: libc::c_long = 167;
    #[cfg(target_arch = "aarch64")]
    const SYS_SWAPON: libc::c_long = 224;
    #[cfg(target_arch = "x86_64")]
    const SYS_SWAPOFF: libc::c_long = 168;
    #[cfg(target_arch = "aarch64")]
    const SYS_SWAPOFF: libc::c_long = 225;
    const SWAP_FLAG_PREFER: libc::c_int = 0x8000;

    fn reset_zram() {
        let _ = std::fs::write("/sys/block/zram0/reset", "1");
    }

    // Wait for the built-in zram device to appear.
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
        log_warn!("[zram] setup failed: device not found");
        return ZramSetupOutcome::Failed;
    }

    let requested_algorithm = "lz4";
    let dev = std::ffi::CString::new("/dev/zram0").unwrap();

    // Configure geometry and activate swap. Every step here is a reset-only
    // failure — swapon is the last fallible step, so any Err leaves swap
    // inactive and a bare reset_zram() is always the correct cleanup. The
    // closure boundary ends exactly at swapon; guard.spawn() and remove_file
    // below keep their own stage-specific cleanup.
    let setup =
        (|| -> Result<(crate::oom_guard::PreparedOomGuard, crate::oom_guard::VerifiedZramGeometry), String> {
            // The fixed geometry and pressure guard require lz4. The kernel rejects
            // unknown algorithms at write time, so a successful write is the selection.
            std::fs::write("/sys/block/zram0/comp_algorithm", requested_algorithm)
                .map_err(|_| "could not set required lz4 compression".to_string())?;

            // Kernel 6.16+: algorithm-specific tuning via sysfs (best-effort).
            let _ = std::fs::write("/sys/block/zram0/algorithm_params", "level=1");

            let (zram_size, mem_limit) = zram_geometry(read_mem_total_kb());

            // Logical swap is 40% of RAM. Compression reduces its physical backing
            // cost; it does not increase this disksize ceiling.
            std::fs::write("/sys/block/zram0/disksize", zram_size.to_string())
                .map_err(|_| "could not set disksize".to_string())?;

            // Cap compressed backing at 20% of RAM. Logical slots remain advertised
            // when incompressible writes fill that cap, so the PSI guard terminates
            // the VM if reclaim then stalls at substantial zram occupancy.
            std::fs::write("/sys/block/zram0/mem_limit", mem_limit.to_string())
                .map_err(|e| format!("could not set mem_limit: {e}"))?;

            // The mem_limit sysfs node is write-only. mm_stat field 4 is the
            // kernel's effective, page-rounded value and is the actual invariant.
            // PSI must also be operational before this geometry may be activated.
            let (oom_guard, geometry) =
                crate::oom_guard::PreparedOomGuard::prepare(mem_limit, zram_size)
                    .map_err(|e| format!("OOM guard verification: {e}"))?;

            // Build and write swap header (mkswap equivalent).
            let header = build_swap_header(geometry.disksize)
                .ok_or_else(|| "device too small for swap".to_string())?;
            std::fs::OpenOptions::new()
                .write(true)
                .open("/dev/zram0")
                .and_then(|mut f| f.write_all(&header))
                .map_err(|e| format!("mkswap: {e}"))?;

            // swapon with high priority. Returning 0 attests activation; no
            // /proc/swaps readback needed.
            if unsafe { libc::syscall(SYS_SWAPON, dev.as_ptr(), SWAP_FLAG_PREFER | 100) } < 0 {
                return Err(format!("swapon: {}", std::io::Error::last_os_error()));
            }
            Ok((oom_guard, geometry))
        })();
    let (oom_guard, geometry) = match setup {
        Ok(v) => v,
        Err(e) => {
            log_warn!("[zram] setup failed: {e}");
            reset_zram();
            return ZramSetupOutcome::Failed;
        }
    };

    // Start only after verified zram activation. If the dedicated thread cannot
    // be created, immediately remove swap and reset zram rather than leave the
    // capped geometry active without its pressure guard.
    if let Err(e) = oom_guard.spawn() {
        log_warn!("[zram] setup failed: could not start OOM guard: {e}");
        let swapoff_rc = unsafe { libc::syscall(SYS_SWAPOFF, dev.as_ptr()) };
        let reset_result = std::fs::write("/sys/block/zram0/reset", "1");
        if swapoff_rc != 0 || reset_result.is_err() {
            log_warn!("[zram] cleanup after guard failure failed, powering off guest");
            crate::oom_guard::emergency_poweroff();
        }
        return ZramSetupOutcome::Failed;
    }

    log_info!(
        "[zram] ready: active {} swap with PSI guard (disksize={}, mem_limit={})",
        requested_algorithm,
        geometry.disksize,
        geometry.mem_limit,
    );

    // Security: remove the setup-only device node before /dev is sealed and
    // before untrusted execution readiness is published.
    if let Err(e) = std::fs::remove_file("/dev/zram0") {
        log_warn!("[zram] failed to remove /dev/zram0: {e}");
        return ZramSetupOutcome::Failed;
    }

    // VM tuning for zram is pre-applied in apply_zram_vm_tuning() before
    // /proc/sys is made read-only. No sysctl writes needed here.
    ZramSetupOutcome::Active
}

/// Build a swap header for the given device size.
/// Returns a PAGE_SIZE-byte header or None if the size is too small.
///
/// The swap header layout is PAGE_SIZE-dependent:
/// - Header occupies one page (PAGE_SIZE bytes)
/// - SWAPSPACE2 signature at offset (PAGE_SIZE - 10)
/// - Page count = device_size / PAGE_SIZE
/// - Version + last_page at fixed offsets 1024 and 1028
///
/// Uses runtime sysconf(_SC_PAGESIZE) to support both 4KB (x86_64)
/// and 16KB (aarch64 with CONFIG_ARM64_16K_PAGES) kernels.
fn build_swap_header(device_size: u64) -> Option<Vec<u8>> {
    let page_size = page_size();
    let pages = (device_size / page_size).saturating_sub(1) as u32;
    if pages < 10 {
        return None; // kernel rejects tiny swap
    }
    let ps = page_size as usize;
    let mut header = vec![0u8; ps];
    // SWAPSPACE2 signature at offset (PAGE_SIZE - 10)
    header[ps - 10..ps].copy_from_slice(b"SWAPSPACE2");
    // version = 1
    header[1024..1028].copy_from_slice(&1u32.to_le_bytes());
    // last_page (0-indexed: total_pages - 1, since page 0 is header)
    header[1028..1032].copy_from_slice(&pages.to_le_bytes());
    Some(header)
}

// ============================================================================
// Sysctl hardening
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

/// E3: Non-critical sysctl — defense-in-depth settings applied synchronously
/// in phase 2 core (before port open). These harden filesystem and limit
/// attack surface but aren't exploitable during early boot.
const SYSCTL_NON_CRITICAL: &[(&str, &str)] = &[
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

fn apply_sysctl_non_critical() {
    for &(path, value) in SYSCTL_NON_CRITICAL {
        let _ = std::fs::write(path, value);
    }
}

/// Pre-apply zram VM tuning sysctls before /proc/sys is made read-only.
/// These values are safe to set even before zram device setup completes:
/// the kernel uses defaults until swapon, then these take effect.
///
/// Refs:
///   - https://docs.kernel.org/admin-guide/sysctl/vm.html
///   - https://wiki.archlinux.org/title/Zram (recommended zram sysctls)
fn apply_zram_vm_tuning() {
    let mem_kb = read_mem_total_kb();

    let _ = std::fs::write("/proc/sys/vm/page-cluster", "0");
    // 100, not the Arch-wiki zram value of 180: that advice assumes an
    // UNCAPPED zram. With mem_limit set, swappiness=180 makes reclaim bang
    // exclusively on the zram wall once the cap is hit (writes fail with the
    // swap still advertised as free), feeding the should_reclaim_retry
    // capped-zram reclaim stall. 100 keeps file-page reclaim in the mix while
    // the PSI/zram guard handles a confirmed allocator-stall condition.
    let _ = std::fs::write("/proc/sys/vm/swappiness", "100");
    let _ = std::fs::write(
        "/proc/sys/vm/min_free_kbytes",
        (mem_kb * 4 / 100).to_string(),
    );
    let _ = std::fs::write("/proc/sys/vm/overcommit_memory", "0");
    // Kill the task that triggered OOM immediately instead of scanning the
    // full task list for the "best" candidate. In a single-user sandbox VM
    // there is only one meaningful process, so heuristic selection wastes
    // time. Ref: https://docs.kernel.org/admin-guide/sysctl/vm.html
    let _ = std::fs::write("/proc/sys/vm/oom_kill_allocating_task", "1");
    // Disable watermark boosting — designed for spinning-disk fragmentation
    // avoidance, counterproductive with zram (causes premature reclaim that
    // wastes CPU on compression). Ref: https://wiki.archlinux.org/title/Zram
    let _ = std::fs::write("/proc/sys/vm/watermark_boost_factor", "0");
    // Widen kswapd wake-up range so background reclaim starts earlier and
    // direct-reclaim stalls are less likely. Default 10 (0.1% of RAM) is
    // too narrow for small VMs. 125 = 1.25% of RAM.
    // Ref: https://wiki.archlinux.org/title/Zram
    let _ = std::fs::write("/proc/sys/vm/watermark_scale_factor", "125");
}

// ============================================================================
// Helpers
// ============================================================================

/// Read MemTotal from /proc/meminfo in kilobytes.
/// Panics if /proc/meminfo is unreadable or MemTotal is missing — we control the
/// environment and /proc is always mounted before any tmpfs. A missing MemTotal
/// means the VM is broken and should not proceed.
///
/// NOTE: Mirrored in tiny-init/src/sys.rs — keep both in sync.
fn read_mem_total_kb() -> u64 {
    let meminfo = std::fs::read_to_string("/proc/meminfo").expect("/proc/meminfo unreadable");
    meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|n| n.parse().ok())
        .expect("MemTotal not found in /proc/meminfo")
}

/// Runtime page size from sysconf(_SC_PAGESIZE).
/// Supports both 4KB (x86_64) and 16KB (aarch64 with CONFIG_ARM64_16K_PAGES).
/// (tiny-init/src/sys.rs keeps its own raw copy for its separate crate.)
fn page_size() -> u64 {
    // Single checked implementation lives in oom_guard; sysconf(_SC_PAGESIZE)
    // cannot fail on Linux, and a PID-1 panic here is a boot failure anyway.
    crate::oom_guard::page_size().expect("sysconf(_SC_PAGESIZE)")
}

/// Verify gvproxy connectivity with exponential backoff.
/// Runs inside the background network worker; ExecuteCode/InstallPackages wait
/// on the untrusted-readiness gate, which reflects this probe's outcome.
///
/// DNS-first verification (no fork/exec):
/// UDP DNS query to gateway:53 — any response proves both L3 routing AND DNS proxy
/// are ready (can't get a UDP response without working L3). We query `_probe.internal`
/// which is guaranteed to hit gvproxy's DNS proxy without external dependencies.
/// Any response (including NXDOMAIN) = ready.
///
/// If DNS fails after all retries, a single TCP probe to gateway:1 distinguishes
/// "no L3" from "L3 ok but DNS broken" — for diagnostics only.
fn verify_gvproxy() -> Result<(), &'static str> {
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
                    return Ok(());
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
        Err("gvproxy DNS probe failed")
    } else {
        log_warn!("gvproxy unreachable (no L3), network will not work");
        Err("gvproxy L3 probe failed")
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

/// Mask a path by bind-mounting /dev/null over it (returns empty on read).
fn mount_mask(path: &std::ffi::CStr) -> bool {
    unsafe {
        let ret = libc::mount(
            c"/dev/null".as_ptr(),
            path.as_ptr(),
            std::ptr::null(),
            libc::MS_BIND,
            std::ptr::null(),
        );
        if ret != 0 {
            log_warn!(
                "mask mount {} failed: {}",
                path.to_string_lossy(),
                std::io::Error::last_os_error()
            );
            return false;
        }
        true
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::ZRAM_PENDING;

    #[test]
    fn network_state_decoder_distinguishes_all_terminal_states() {
        assert_eq!(decode_network_state(NETWORK_PENDING), Ok(false));
        assert_eq!(decode_network_state(NETWORK_READY), Ok(true));
        assert_eq!(
            decode_network_state(NETWORK_FAILED),
            Err("network setup failed")
        );
    }

    #[test]
    fn network_state_decoder_rejects_unknown_values() {
        assert!(decode_network_state(200).is_err());
    }

    #[test]
    fn zram_geometry_is_40_percent_disk_20_percent_mem() {
        // 1000 KiB => 1_024_000 bytes; independent literals, not the formula.
        assert_eq!(zram_geometry(1000), (409_600, 204_800));
        assert_eq!(zram_geometry(0), (0, 0));
    }

    #[test]
    fn swap_header_layout_matches_swapspace2() {
        let ps = page_size() as usize;

        // Kernel rejects tiny swap: 10 usable pages is the floor.
        assert!(build_swap_header((10 * ps) as u64).is_none());

        let header = build_swap_header((12 * ps) as u64).expect("12 pages is enough");
        assert_eq!(header.len(), ps);
        assert_eq!(&header[ps - 10..], b"SWAPSPACE2");
        assert_eq!(&header[1024..1028], &1u32.to_le_bytes()); // version
        assert_eq!(&header[1028..1032], &11u32.to_le_bytes()); // last_page = 12 - 1
    }

    #[tokio::test]
    async fn untrusted_readiness_fails_closed_on_zram_before_network() {
        // NETWORK_STATE stays PENDING: if the zram check did not run first,
        // this call would block on the network waiter instead of erroring.
        ZRAM_STATE.store(ZRAM_FAILED, Ordering::Release);
        let result = wait_for_untrusted_readiness().await;
        assert_eq!(result, Err("zram safety setup failed"));
        ZRAM_STATE.store(ZRAM_PENDING, Ordering::Release);
    }

    #[test]
    fn zram_state_decoder_fails_closed_on_anything_but_active() {
        assert_eq!(decode_zram_state(ZRAM_READY), Ok(()));
        assert_eq!(
            decode_zram_state(ZRAM_FAILED),
            Err("zram safety setup failed")
        );
        assert_eq!(
            decode_zram_state(ZRAM_PENDING),
            Err("zram safety state is not terminal")
        );
    }
}

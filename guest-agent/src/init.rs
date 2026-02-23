//! Init-time setup: PID 1 environment, mounts, network, zombie reaping.

use std::process::Command as StdCommand;

use crate::constants::{SANDBOX_GID, SANDBOX_UID};

// ============================================================================
// Zombie reaping
// ============================================================================

/// Reap zombie processes when running as PID 1.
pub(crate) async fn reap_zombies() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigchld = match signal(SignalKind::child()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: Failed to register SIGCHLD handler: {e}");
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

/// Handles userspace-only initialization after tiny-init.
pub(crate) fn setup_init_environment() {
    setup_env_and_permissions();
    mount_home_tmpfs();
    mount_readonly_paths();
    setup_network();
}

/// Set PATH, UV_NO_CACHE, and harden directory permissions.
fn setup_env_and_permissions() {
    // SAFETY: called at startup before any threads are spawned
    unsafe { std::env::set_var("PATH", "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin") };
    eprintln!("Set PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin");

    unsafe { std::env::set_var("UV_NO_CACHE", "1") };
    eprintln!("Set UV_NO_CACHE=1");

    // CIS Benchmark 6.1.x compliance
    chmod_paths("755", &["/etc", "/usr", "/var", "/sbin", "/bin"]);
    chmod_paths(
        "644",
        &[
            "/etc/passwd",
            "/etc/group",
            "/etc/resolv.conf",
            "/etc/hosts",
        ],
    );
    chmod_paths("640", &["/etc/shadow"]);
    eprintln!("Hardened system directory permissions (CIS 6.1.x)");
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
    if ret == 0 {
        eprintln!("Mounted tmpfs on /home/user (nosuid,nodev)");
    } else {
        eprintln!(
            "Warning: tmpfs mount on /home/user failed: {}",
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
        eprintln!("init.rw=1: skipping /usr, /bin, /sbin read-only remounts");
    } else {
        for (path, label) in [
            (c"/usr", "/usr read-only with nosuid"),
            (c"/bin", "/bin read-only with nosuid"),
            (c"/sbin", "/sbin read-only with nosuid"),
        ] {
            if mount_readonly(path) {
                eprintln!("Mounted {label}");
            }
        }
    }

    for (path, label) in [
        (c"/etc/hosts", "/etc/hosts read-only"),
        (c"/etc/resolv.conf", "/etc/resolv.conf read-only"),
        (c"/dev", "/dev read-only with nosuid"),
        (c"/proc/sys", "/proc/sys read-only"),
        (c"/proc/sysrq-trigger", "/proc/sysrq-trigger read-only"),
    ] {
        if mount_readonly(path) {
            eprintln!("Mounted {label}");
        }
    }
}

/// Configure loopback and eth0 network interfaces.
fn setup_network() {
    let _ = StdCommand::new("ip")
        .args(["link", "set", "lo", "up"])
        .status();
    eprintln!("Loopback interface up");

    // Wait for eth0 (up to 1 second)
    for _ in 0..50 {
        if std::path::Path::new("/sys/class/net/eth0").exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    if std::path::Path::new("/sys/class/net/eth0").exists() {
        eprintln!("Configuring network...");
        let _ = StdCommand::new("ip")
            .args(["link", "set", "eth0", "up"])
            .status();
        let _ = StdCommand::new("ip")
            .args(["addr", "add", "192.168.127.2/24", "dev", "eth0"])
            .status();
        let _ = StdCommand::new("ip")
            .args(["route", "add", "default", "via", "192.168.127.1"])
            .status();
        eprintln!("Network configured: 192.168.127.2/24 via 192.168.127.1");

        // Verify gvproxy connectivity
        let mut gvproxy_ok = false;
        for delay_ms in [50, 100, 200, 400, 800, 1000, 1000] {
            let result = StdCommand::new("ping")
                .args(["-c", "1", "-W", "1", "192.168.127.1"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            if let Ok(status) = result
                && status.success()
            {
                eprintln!("gvproxy connectivity verified");
                gvproxy_ok = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }
        if !gvproxy_ok {
            eprintln!("Warning: gvproxy not reachable, package install may fail");
        }
    } else {
        eprintln!("Warning: eth0 not found, network unavailable");
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Set file mode on multiple paths (non-fatal).
fn chmod_paths(mode: &str, paths: &[&str]) {
    for path in paths {
        let _ = StdCommand::new("chmod").args([mode, path]).status();
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
            eprintln!(
                "Warning: bind mount {} failed: {}",
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
            eprintln!(
                "Warning: RO remount {} failed: {}",
                path.to_string_lossy(),
                std::io::Error::last_os_error()
            );
            return false;
        }
        true
    }
}

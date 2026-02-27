//! Syscall wrappers, constants, and kernel queries.

use std::ffi::CString;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;

// Mount flags
pub(crate) const MS_NOSUID: libc::c_ulong = 0x2;
pub(crate) const MS_NODEV: libc::c_ulong = 0x4;
pub(crate) const MS_RDONLY: libc::c_ulong = 0x1;
pub(crate) const MS_NOATIME: libc::c_ulong = 0x400;
pub(crate) const MS_BIND: libc::c_ulong = 0x1000;
pub(crate) const MS_MOVE: libc::c_ulong = 0x2000;

/// Write to stderr using raw syscall (safe before Rust stdio init).
///
/// # Why not eprintln!?
///
/// When running as PID 1 (init) in an initramfs, Rust's `std::io::stderr()`
/// causes SIGSEGV because:
/// - Lazy initialization of stdio may access invalid memory before console setup
/// - Internal mutexes/TLS may not be properly initialized
/// - File descriptors 0/1/2 may not exist yet
///
/// This macro uses raw `libc::write(fd=2)` which:
/// - Is a direct syscall with no Rust runtime dependencies
/// - Returns -1 (EBADF) instead of crashing if fd 2 doesn't exist
/// - Works immediately, even before /dev is mounted
///
/// See: https://github.com/rust-lang/rust/issues/24821
/// See: https://rust-lang.github.io/rfcs/1014-stdout-existential-crisis.html
macro_rules! _log_write {
    ($level:literal, $($arg:tt)*) => {{
        let msg = format!(concat!("[", $level, "] {}\n"), format_args!($($arg)*));
        unsafe {
            libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
        }
    }};
}

macro_rules! log_info {
    ($($arg:tt)*) => { _log_write!("INFO", $($arg)*) };
}

macro_rules! log_warn {
    ($($arg:tt)*) => { _log_write!("WARN", $($arg)*) };
}

macro_rules! log_error {
    ($($arg:tt)*) => { _log_write!("ERROR", $($arg)*) };
}

/// Get monotonic time in microseconds (for boot timing instrumentation).
pub(crate) fn monotonic_us() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1_000_000 + (ts.tv_nsec as u64) / 1_000
}

pub(crate) fn last_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}

pub(crate) fn mount(
    source: &str,
    target: &str,
    fstype: &str,
    flags: libc::c_ulong,
    data: &str,
) -> i32 {
    let source = CString::new(source).unwrap();
    let target = CString::new(target).unwrap();
    let fstype = CString::new(fstype).unwrap();
    let data = CString::new(data).unwrap();

    unsafe {
        libc::mount(
            source.as_ptr(),
            target.as_ptr(),
            fstype.as_ptr(),
            flags,
            data.as_ptr() as *const libc::c_void,
        )
    }
}

pub(crate) fn mount_move(source: &str, target: &str) -> i32 {
    let source = CString::new(source).unwrap();
    let target = CString::new(target).unwrap();

    unsafe {
        libc::mount(
            source.as_ptr(),
            target.as_ptr(),
            std::ptr::null(),
            MS_MOVE,
            std::ptr::null(),
        )
    }
}

pub(crate) fn umount(target: &str) -> i32 {
    let target = CString::new(target).unwrap();
    unsafe { libc::umount(target.as_ptr()) }
}

fn parse_cmdline_has(cmdline: &str, flag: &str) -> bool {
    cmdline.split_whitespace().any(|arg| arg == flag)
}

pub(crate) fn cmdline_has(flag: &str) -> bool {
    fs::read_to_string("/proc/cmdline")
        .map(|s| parse_cmdline_has(&s, flag))
        .unwrap_or(false)
}

/// Read MemTotal from /proc/meminfo in kilobytes.
/// Panics if /proc/meminfo is unreadable or MemTotal is missing — /proc is
/// always mounted before this is called, and a missing MemTotal means the VM
/// is broken and should not proceed.
///
/// NOTE: Mirrored in guest-agent/src/init.rs — keep both in sync.
pub(crate) fn read_mem_total_kb() -> u64 {
    let meminfo = fs::read_to_string("/proc/meminfo").expect("/proc/meminfo unreadable");
    meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|n| n.parse().ok())
        .expect("MemTotal not found in /proc/meminfo")
}

pub(crate) fn fallback_shell() -> ! {
    // exec /bin/sh (or sleep forever if no shell)
    for shell in ["/bin/sh", "/bin/ash"] {
        if Path::new(shell).exists() {
            let prog = CString::new(shell).unwrap();
            let args: [*const libc::c_char; 2] = [prog.as_ptr(), std::ptr::null()];
            unsafe { libc::execv(prog.as_ptr(), args.as_ptr()) };
        }
    }
    loop {
        thread::sleep(Duration::from_secs(3600));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serial_test::serial;

    #[test]
    fn cmdline_parse_present() {
        assert!(parse_cmdline_has(
            "console=ttyS0 init.net=1 quiet",
            "init.net=1"
        ));
    }

    #[test]
    fn cmdline_parse_absent() {
        assert!(!parse_cmdline_has(
            "console=ttyS0 init.net=1 quiet",
            "init.net=0"
        ));
    }

    #[test]
    fn cmdline_parse_empty() {
        assert!(!parse_cmdline_has("", "init.net=1"));
    }

    #[test]
    fn cmdline_parse_no_partial_match() {
        // "init.net=10" must NOT match "init.net=1"
        assert!(!parse_cmdline_has("init.net=10", "init.net=1"));
    }

    #[test]
    fn cmdline_parse_whitespace_variations() {
        assert!(parse_cmdline_has("  init.rw=1  ", "init.rw=1"));
        assert!(parse_cmdline_has("a\tinit.rw=1\tb", "init.rw=1"));
    }

    #[test]
    fn cmdline_parse_bare_flag() {
        assert!(parse_cmdline_has("quiet nosplash debug", "quiet"));
        assert!(!parse_cmdline_has("quiet nosplash debug", "quie"));
    }

    #[test]
    fn cmdline_parse_duplicate_flags() {
        assert!(parse_cmdline_has("init.net=1 init.net=1", "init.net=1"));
    }

    proptest! {
        #[test]
        fn cmdline_never_partial_matches(
            flag in "[a-z][a-z.=0-9]{1,20}",
            suffix in "[a-z0-9]{1,5}",
        ) {
            // A flag with extra suffix chars should NOT match
            let padded = format!("{}{}", flag, suffix);
            prop_assert!(!parse_cmdline_has(&padded, &flag));
        }

        #[test]
        fn cmdline_exact_match_always_works(
            flag in "[a-z][a-z.=0-9]{1,20}",
            padding in "[a-z ]{0,10}",
        ) {
            let cmdline = format!("{} {} other", padding, flag);
            prop_assert!(parse_cmdline_has(&cmdline, &flag));
        }
    }

    #[test]
    #[serial]
    fn last_errno_after_failed_open() {
        let bad = CString::new("/nonexistent_path_xxxxx").unwrap();
        unsafe { libc::open(bad.as_ptr(), libc::O_RDONLY) };
        assert_eq!(last_errno(), 2); // ENOENT
    }

    #[test]
    #[serial]
    fn last_errno_after_failed_chmod() {
        let bad = CString::new("/nonexistent_path_xxxxx").unwrap();
        unsafe { libc::chmod(bad.as_ptr(), 0) };
        assert_eq!(last_errno(), 2); // ENOENT
    }
}

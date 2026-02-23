//! Device polling, console management, and device node operations.

use std::ffi::CString;
use std::fs;
use std::os::unix::fs::symlink;
use std::thread;
use std::time::Duration;

pub(crate) const CONSOLE_DEVICES: &[&str] = &["/dev/hvc0", "/dev/ttyS0", "/dev/ttyAMA0"];
pub(crate) const CONSOLE_DEVICES_WITH_FALLBACK: &[&str] =
    &["/dev/hvc0", "/dev/ttyAMA0", "/dev/ttyS0", "/dev/console"];

pub(crate) fn poll_backoff(delays_us: &[u64], mut check: impl FnMut() -> bool) -> bool {
    for &delay_us in delays_us {
        if check() {
            return true;
        }
        thread::sleep(Duration::from_micros(delay_us));
    }
    check() // final check after last sleep
}

pub(crate) fn open_console(devices: &[&str], mode: libc::c_int, fds: &[libc::c_int]) -> bool {
    for &console in devices {
        if let Ok(path) = CString::new(console) {
            let fd = unsafe { libc::open(path.as_ptr(), mode) };
            if fd >= 0 {
                unsafe {
                    for &target_fd in fds {
                        libc::dup2(fd, target_fd);
                    }
                    if fd > 2 {
                        libc::close(fd);
                    }
                }
                return true;
            }
        }
    }
    false
}

pub(crate) fn redirect_to_console() {
    // Redirect stdout/stderr to console device
    // Directly try to open each device (avoids TOCTOU race)
    // hvc0: virtio-console (microvm, virt with virtio-console)
    // ttyS0: x86 serial (pc machine, TCG)
    // ttyAMA0: ARM64 PL011 UART (virt machine)
    open_console(CONSOLE_DEVICES, libc::O_WRONLY, &[1, 2]);
}

pub(crate) fn wait_for_block_device(path: &str) -> bool {
    // Wait for block device by attempting to open it (avoids TOCTOU race)
    // O_RDONLY | O_NONBLOCK: non-blocking open to check device availability
    let path_cstr = match CString::new(path) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // B3: Tighter exponential backoff: 0.2+0.5+1+2+4+8+16 = 31.7ms max
    poll_backoff(&[200, 500, 1000, 2000, 4000, 8000, 16000], || {
        let fd = unsafe { libc::open(path_cstr.as_ptr(), libc::O_RDONLY | libc::O_NONBLOCK) };
        if fd >= 0 {
            unsafe { libc::close(fd) };
            true
        } else {
            false
        }
    })
}

pub(crate) fn wait_for_virtio_ports() -> bool {
    // Wait for virtio-ports directory to have entries
    // Uses any() to stop at first entry (more efficient than count())
    // B3: Tighter exponential backoff: 0.2+0.5+1+2+4+8+16 = 31.7ms max
    poll_backoff(&[200, 500, 1000, 2000, 4000, 8000, 16000], || {
        fs::read_dir("/sys/class/virtio-ports")
            .map(|mut entries| entries.any(|e| e.is_ok()))
            .unwrap_or(false)
    })
}

pub(crate) fn setup_virtio_ports() {
    // mkdir -p /dev/virtio-ports
    let _ = fs::create_dir_all("/dev/virtio-ports");

    // for vport in /sys/class/virtio-ports/vport*
    let entries = match fs::read_dir("/sys/class/virtio-ports") {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("vport") {
            continue;
        }

        // port_name=$(cat "$vport/name")
        if let Ok(port_name) = fs::read_to_string(entry.path().join("name")) {
            let port_name = port_name.trim();
            if !port_name.is_empty() {
                // ln -sf "../$dev_name" "/dev/virtio-ports/$port_name"
                let _ = symlink(
                    format!("../{}", name_str),
                    format!("/dev/virtio-ports/{}", port_name),
                );
            }
        }
    }
}

pub(crate) fn remove_device_node(path: &str) {
    // Try to remove the device node entirely
    if fs::remove_file(path).is_ok() {
        return;
    }
    // Fallback: chmod 000 blocks UID 1000 access (CAP_DAC_OVERRIDE can bypass,
    // but user code runs as UID 1000). Effective on devtmpfs.
    if let Ok(cpath) = CString::new(path)
        && unsafe { libc::chmod(cpath.as_ptr(), 0) } == 0
    {
        log_fmt!(
            "[init] WARNING: could not remove {}, chmod 000 applied",
            path
        );
        return;
    }
    log_fmt!("[init] WARNING: could not remove or chmod {}", path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serial_test::serial;

    // poll_backoff tests

    #[test]
    fn poll_backoff_immediate_success() {
        assert!(poll_backoff(&[0, 0], || true));
    }

    #[test]
    fn poll_backoff_delayed_success() {
        let mut count = 0;
        assert!(poll_backoff(&[0, 0, 0], || {
            count += 1;
            count >= 3
        }));
    }

    #[test]
    fn poll_backoff_total_failure() {
        assert!(!poll_backoff(&[0, 0], || false));
    }

    #[test]
    fn poll_backoff_empty_delays() {
        // Just the final check
        assert!(poll_backoff(&[], || true));
        assert!(!poll_backoff(&[], || false));
    }

    #[test]
    fn poll_backoff_final_check_matters() {
        // Fails in-loop, succeeds on final check after last sleep
        let mut calls = 0;
        let delays = [0, 0, 0];
        assert!(poll_backoff(&delays, || {
            calls += 1;
            calls == 4
        }));
    }

    #[test]
    fn poll_backoff_check_count() {
        // Exact call count: len(delays) in-loop + 1 final
        let mut count = 0;
        poll_backoff(&[0, 0, 0], || {
            count += 1;
            false
        });
        assert_eq!(count, 4);
    }

    #[test]
    fn poll_backoff_short_circuits_on_success() {
        let mut count = 0;
        poll_backoff(&[0, 0, 0, 0, 0], || {
            count += 1;
            count == 2
        });
        assert_eq!(count, 2); // stopped after 2nd check
    }

    #[test]
    fn poll_backoff_single_delay() {
        let mut count = 0;
        assert!(!poll_backoff(&[0], || {
            count += 1;
            false
        }));
        assert_eq!(count, 2); // 1 in-loop + 1 final
    }

    proptest! {
        #[test]
        fn poll_backoff_succeeds_at_correct_index(
            n_delays in 0..30usize,
            success_at in 0..32usize,
        ) {
            let delays: Vec<u64> = vec![0; n_delays];
            let mut calls = 0;
            let result = poll_backoff(&delays, || {
                calls += 1;
                calls > success_at
            });
            // Total checks = n_delays (in-loop) + 1 (final) = n_delays + 1
            // Succeeds if success_at < n_delays + 1, i.e. success_at <= n_delays
            prop_assert_eq!(result, success_at <= n_delays);
        }
    }

    // Console device constant tests

    #[test]
    fn console_devices_all_start_with_dev() {
        for &dev in CONSOLE_DEVICES.iter().chain(CONSOLE_DEVICES_WITH_FALLBACK) {
            assert!(dev.starts_with("/dev/"), "invalid device: {}", dev);
        }
    }

    #[test]
    fn console_fallback_includes_dev_console() {
        assert!(CONSOLE_DEVICES_WITH_FALLBACK.contains(&"/dev/console"));
    }

    #[test]
    fn console_devices_not_empty() {
        assert!(!CONSOLE_DEVICES.is_empty());
        assert!(!CONSOLE_DEVICES_WITH_FALLBACK.is_empty());
    }

    #[test]
    fn console_devices_no_null_bytes() {
        // CString::new would fail on null bytes
        for &dev in CONSOLE_DEVICES.iter().chain(CONSOLE_DEVICES_WITH_FALLBACK) {
            assert!(!dev.contains('\0'), "null byte in device path: {}", dev);
        }
    }

    // open_console tests

    #[test]
    fn open_console_empty_list_returns_false() {
        assert!(!open_console(&[], libc::O_WRONLY, &[1, 2]));
    }

    #[test]
    fn open_console_nonexistent_returns_false() {
        assert!(!open_console(
            &["/dev/nonexistent_xxx"],
            libc::O_WRONLY,
            &[1, 2]
        ));
    }

    #[test]
    #[serial] // mutates fd 1
    fn open_console_with_temp_file() {
        let tmp = std::env::temp_dir().join("tiny-init-test-console");
        std::fs::write(&tmp, "").unwrap();
        let path_str = tmp.to_str().unwrap();

        // Save stdout before mutation
        let saved = unsafe { libc::dup(1) };
        assert!(saved >= 0, "dup(1) failed");

        let result = open_console(&[path_str], libc::O_WRONLY, &[1]);

        // Restore stdout and clean up BEFORE asserting (prevents fd/file leak on panic)
        unsafe {
            libc::dup2(saved, 1);
            libc::close(saved);
        }
        let _ = std::fs::remove_file(&tmp);
        assert!(result);
    }

    #[test]
    #[serial] // mutates fd 1
    fn open_console_skips_bad_then_finds_good() {
        let tmp = std::env::temp_dir().join("tiny-init-test-console2");
        std::fs::write(&tmp, "").unwrap();
        let path_str = tmp.to_str().unwrap();

        let saved = unsafe { libc::dup(1) };
        let result = open_console(&["/dev/nonexistent_xxx", path_str], libc::O_WRONLY, &[1]);

        // Restore stdout and clean up BEFORE asserting
        unsafe {
            libc::dup2(saved, 1);
            libc::close(saved);
        }
        let _ = std::fs::remove_file(&tmp);
        assert!(result);
    }

    #[test]
    fn open_console_empty_fds_slice() {
        // No fds to redirect — should open device but dup nothing
        let tmp = std::env::temp_dir().join("tiny-init-test-console3");
        std::fs::write(&tmp, "").unwrap();
        let path_str = tmp.to_str().unwrap();

        assert!(open_console(&[path_str], libc::O_WRONLY, &[]));
        std::fs::remove_file(&tmp).unwrap();
    }
}

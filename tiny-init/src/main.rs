//! Minimal init for QEMU microVMs
//!
//! Pure Rust port of images/minimal-init.sh - no busybox dependency.

use std::ffi::CString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::unix::fs::symlink;
use std::path::Path;
use std::thread;
use std::time::Duration;

// Syscall numbers
#[cfg(target_arch = "x86_64")]
mod syscall_nr {
    pub const INIT_MODULE: libc::c_long = 175;
    pub const SWAPON: libc::c_long = 167;
}

#[cfg(target_arch = "aarch64")]
mod syscall_nr {
    pub const INIT_MODULE: libc::c_long = 105;
    pub const SWAPON: libc::c_long = 224;
}

// Mount flags
const MS_MOVE: libc::c_ulong = 0x2000;
const MS_NOATIME: libc::c_ulong = 0x400;

// Swap flags
const SWAP_FLAG_PREFER: libc::c_int = 0x8000;

fn mount(source: &str, target: &str, fstype: &str, flags: libc::c_ulong, data: &str) -> i32 {
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

fn mount_move(source: &str, target: &str) -> i32 {
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

fn load_module(path: &str) {
    // Read uncompressed module directly (modules decompressed at build time)
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return,
    };

    let mut data = Vec::new();
    if file.read_to_end(&mut data).is_err() {
        return;
    }

    let params = CString::new("").unwrap();
    // insmod - ignore errors (2>/dev/null)
    unsafe {
        libc::syscall(
            syscall_nr::INIT_MODULE,
            data.as_ptr(),
            data.len(),
            params.as_ptr(),
        );
    }
}

fn get_kernel_version() -> Option<String> {
    let mut utsname: libc::utsname = unsafe { std::mem::zeroed() };
    if unsafe { libc::uname(&mut utsname) } != 0 {
        return None;
    }
    Some(
        unsafe { std::ffi::CStr::from_ptr(utsname.release.as_ptr()) }
            .to_string_lossy()
            .into_owned(),
    )
}

fn cmdline_has(flag: &str) -> bool {
    fs::read_to_string("/proc/cmdline")
        .map(|s| s.split_whitespace().any(|arg| arg == flag))
        .unwrap_or(false)
}

fn wait_for_block_device(path: &str) -> bool {
    // Wait for block device by attempting to open it (avoids TOCTOU race)
    // O_RDONLY | O_NONBLOCK: non-blocking open to check device availability
    let path_cstr = match CString::new(path) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // Fast exponential backoff: 1+2+4+8+16+32 = 63ms max (was 155ms)
    for delay_us in [1000, 2000, 4000, 8000, 16000, 32000] {
        let fd = unsafe { libc::open(path_cstr.as_ptr(), libc::O_RDONLY | libc::O_NONBLOCK) };
        if fd >= 0 {
            unsafe { libc::close(fd) };
            return true;
        }
        thread::sleep(Duration::from_micros(delay_us));
    }
    false
}

fn wait_for_virtio_ports() -> bool {
    // Wait for virtio-ports directory to have entries
    // Uses any() to stop at first entry (more efficient than count())
    // Fast exponential backoff: 1+2+4+8+16+32 = 63ms max (was 155ms)
    for delay_us in [1000, 2000, 4000, 8000, 16000, 32000] {
        if let Ok(mut entries) = fs::read_dir("/sys/class/virtio-ports") {
            if entries.any(|e| e.is_ok()) {
                return true;
            }
        }
        thread::sleep(Duration::from_micros(delay_us));
    }
    false
}

fn setup_zram(kver: &str) {
    let m = format!("/lib/modules/{}/kernel", kver);
    load_module(&format!("{}/lib/lz4/lz4_compress.ko", m));
    load_module(&format!("{}/crypto/lz4.ko", m));
    load_module(&format!("{}/drivers/block/zram/zram.ko", m));

    // Wait briefly for zram device to appear (fixes race condition)
    // 20 iterations × 1ms = 20ms max (was 50ms)
    for _ in 0..20 {
        if Path::new("/sys/block/zram0").exists() {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }

    if !Path::new("/sys/block/zram0").exists() {
        return;
    }

    // Use proper fallback chain: lz4 -> lzo-rle -> lzo
    let algorithms = ["lz4", "lzo-rle", "lzo"];
    let mut algo_set = false;
    for algo in algorithms {
        if fs::write("/sys/block/zram0/comp_algorithm", algo).is_ok() {
            algo_set = true;
            break;
        }
    }
    if !algo_set {
        return;
    }

    // MEM_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo)
    let mem_kb: u64 = fs::read_to_string("/proc/meminfo")
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

    // ZRAM_SIZE=$((MEM_KB * 512))
    let zram_size = mem_kb * 512;
    if fs::write("/sys/block/zram0/disksize", zram_size.to_string()).is_err() {
        return;
    }

    // mkswap /dev/zram0 - write swap signature
    // No sync needed - zram is memory-backed
    let header_result = (|| -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new().write(true).open("/dev/zram0")?;
        let mut header = vec![0u8; 4096];
        // SWAPSPACE2 signature at offset 4086
        header[4086..4096].copy_from_slice(b"SWAPSPACE2");
        // version = 1 (write as u32)
        header[1024..1028].copy_from_slice(&1u32.to_le_bytes());
        // last_page
        let pages = (zram_size / 4096) as u32;
        header[1028..1032].copy_from_slice(&pages.to_le_bytes());
        f.write_all(&header)
    })();

    if header_result.is_err() {
        return;
    }

    // swapon -p 100 /dev/zram0
    let dev = CString::new("/dev/zram0").unwrap();
    let ret = unsafe { libc::syscall(syscall_nr::SWAPON, dev.as_ptr(), SWAP_FLAG_PREFER | 100) };
    if ret < 0 {
        return;
    }

    // VM tuning (these can fail silently - non-critical)
    let _ = fs::write("/proc/sys/vm/page-cluster", "0");
    let _ = fs::write("/proc/sys/vm/swappiness", "180");
    let _ = fs::write(
        "/proc/sys/vm/min_free_kbytes",
        (mem_kb * 4 / 100).to_string(),
    );
    let _ = fs::write("/proc/sys/vm/overcommit_memory", "0");
}

fn setup_virtio_ports() {
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

fn redirect_to_console() {
    // Redirect stdout/stderr to console device
    // Directly try to open each device (avoids TOCTOU race)
    // hvc0: virtio-console (microvm, virt with virtio-console)
    // ttyS0: x86 serial (pc machine, TCG)
    // ttyAMA0: ARM64 PL011 UART (virt machine)
    for console in ["/dev/hvc0", "/dev/ttyS0", "/dev/ttyAMA0"] {
        if let Ok(path) = CString::new(console) {
            let fd = unsafe { libc::open(path.as_ptr(), libc::O_WRONLY) };
            if fd >= 0 {
                unsafe {
                    libc::dup2(fd, 1); // stdout
                    libc::dup2(fd, 2); // stderr
                    if fd > 2 {
                        libc::close(fd);
                    }
                }
                return;
            }
        }
    }
}

fn error(msg: &str) {
    eprintln!("[init] ERROR: {}", msg);
}

fn fallback_shell() -> ! {
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

fn switch_root() -> ! {
    // cd /mnt
    if std::env::set_current_dir("/mnt").is_err() {
        error("chdir /mnt failed");
        fallback_shell();
    }

    // mount --move /dev dev etc
    mount_move("/dev", "dev");
    mount_move("/proc", "proc");
    mount_move("/sys", "sys");
    mount_move("/tmp", "tmp");

    // switch_root algorithm (from busybox):
    // Since initramfs IS rootfs, pivot_root cannot work. Instead:
    // 1. mount --move . / (overmount rootfs with new root)
    // 2. chroot .
    // 3. chdir /
    // 4. exec new init
    // See: https://docs.kernel.org/filesystems/ramfs-rootfs-initramfs.html

    let dot = CString::new(".").unwrap();
    let root = CString::new("/").unwrap();

    // Step 1: mount --move . / (overmount rootfs)
    unsafe {
        libc::mount(
            dot.as_ptr(),
            root.as_ptr(),
            std::ptr::null(),
            MS_MOVE,
            std::ptr::null(),
        );
    }

    // Step 2: chroot .
    let chroot_ret = unsafe { libc::chroot(dot.as_ptr()) };
    if chroot_ret != 0 {
        error("chroot failed");
        fallback_shell();
    }

    // Step 3: chdir /
    unsafe {
        libc::chdir(root.as_ptr());
    }

    // Set minimal environment for guest-agent
    std::env::set_var("PATH", "/usr/local/bin:/usr/bin:/bin");
    std::env::set_var("HOME", "/root");

    // Ensure stdin is valid (open /dev/null if needed)
    let devnull = CString::new("/dev/null").unwrap();
    let stdin_fd = unsafe { libc::open(devnull.as_ptr(), libc::O_RDONLY) };
    if stdin_fd >= 0 && stdin_fd != 0 {
        unsafe {
            libc::dup2(stdin_fd, 0);
            libc::close(stdin_fd);
        }
    }

    // Verify stdout/stderr are valid
    let stdout_valid = unsafe { libc::fcntl(1, libc::F_GETFD) } >= 0;
    let stderr_valid = unsafe { libc::fcntl(2, libc::F_GETFD) } >= 0;

    // If stdout/stderr invalid, redirect to console (directly try open, avoids TOCTOU)
    if !stdout_valid || !stderr_valid {
        for console in ["/dev/hvc0", "/dev/ttyAMA0", "/dev/ttyS0", "/dev/console"] {
            if let Ok(path) = CString::new(console) {
                let fd = unsafe { libc::open(path.as_ptr(), libc::O_WRONLY) };
                if fd >= 0 {
                    if !stdout_valid {
                        unsafe {
                            libc::dup2(fd, 1);
                        }
                    }
                    if !stderr_valid {
                        unsafe {
                            libc::dup2(fd, 2);
                        }
                    }
                    if fd > 2 {
                        unsafe {
                            libc::close(fd);
                        }
                    }
                    break;
                }
            }
        }
    }

    // Redirect console for guest-agent (directly try open, avoids TOCTOU)
    // Order: hvc0 (virtio-console), ttyS0 (ISA serial), ttyAMA0 (ARM PL011 UART)
    // vm_manager uses hvc0 for x86 and ttyAMA0 for ARM64
    for console in ["/dev/hvc0", "/dev/ttyS0", "/dev/ttyAMA0"] {
        if let Ok(path) = CString::new(console) {
            let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR) };
            if fd >= 0 {
                unsafe {
                    libc::dup2(fd, 0); // stdin
                    libc::dup2(fd, 1); // stdout
                    libc::dup2(fd, 2); // stderr
                    if fd > 2 {
                        libc::close(fd);
                    }
                }
                break;
            }
        }
    }

    // exec /usr/local/bin/guest-agent
    let prog = CString::new("/usr/local/bin/guest-agent").unwrap();
    let args: [*const libc::c_char; 2] = [prog.as_ptr(), std::ptr::null()];
    unsafe { libc::execv(prog.as_ptr(), args.as_ptr()) };

    error("execv guest-agent failed");
    fallback_shell();
}

fn main() {
    // Mount virtual filesystems
    mount("devtmpfs", "/dev", "devtmpfs", 0, "");
    mount("proc", "/proc", "proc", 0, "");
    mount("sysfs", "/sys", "sysfs", 0, "");
    mount("tmpfs", "/tmp", "tmpfs", 0, "size=128M");

    // Redirect stdout/stderr early (so errors are visible)
    redirect_to_console();

    // Get kernel version
    let kver = match get_kernel_version() {
        Some(v) => v,
        None => {
            error("uname failed");
            fallback_shell();
        }
    };

    // Load modules - check cmdline for optional modules
    let m = format!("/lib/modules/{}/kernel", kver);
    let need_net = cmdline_has("init.net=1");
    let need_balloon = cmdline_has("init.balloon=1");

    // Core virtio (always needed)
    load_module(&format!("{}/drivers/virtio/virtio_mmio.ko", m));
    load_module(&format!("{}/drivers/block/virtio_blk.ko", m));

    // Network modules (only if init.net=1)
    if need_net {
        load_module(&format!("{}/net/core/failover.ko", m));
        load_module(&format!("{}/drivers/net/net_failover.ko", m));
        load_module(&format!("{}/drivers/net/virtio_net.ko", m));
    }

    // Balloon (only if init.balloon=1)
    if need_balloon {
        load_module(&format!("{}/drivers/virtio/virtio_balloon.ko", m));
    }

    // Filesystem modules (always needed)
    load_module(&format!("{}/lib/crc16.ko", m));
    load_module(&format!("{}/crypto/crc32c_generic.ko", m));
    load_module(&format!("{}/lib/libcrc32c.ko", m));
    load_module(&format!("{}/fs/mbcache.ko", m));
    load_module(&format!("{}/fs/jbd2/jbd2.ko", m));
    load_module(&format!("{}/fs/ext4/ext4.ko", m));

    // Setup zram swap
    setup_zram(&kver);

    // Wait for /dev/vda
    if !wait_for_block_device("/dev/vda") {
        error("timeout waiting for /dev/vda");
    }

    // Wait for virtio-serial ports
    wait_for_virtio_ports();

    // Create virtio-ports symlinks
    setup_virtio_ports();

    // Mount root filesystem: mount -t ext4 -o rw,noatime /dev/vda /mnt
    if mount("/dev/vda", "/mnt", "ext4", MS_NOATIME, "") != 0 {
        // Fallback: try without specifying fstype
        if mount("/dev/vda", "/mnt", "", MS_NOATIME, "") != 0 {
            error("mount /dev/vda failed");
            fallback_shell();
        }
    }

    // No existence check - execv will fail if guest-agent missing
    switch_root();
}

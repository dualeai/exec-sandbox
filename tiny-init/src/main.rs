//! Minimal init for QEMU microVMs
//!
//! Pure Rust port of images/minimal-init.sh - no busybox dependency.

use std::ffi::CString;
use std::fs;
use std::io::Write;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::thread;
use std::time::Duration;

// Syscall numbers
#[cfg(target_arch = "x86_64")]
mod syscall_nr {
    pub const FINIT_MODULE: libc::c_long = 313;
    pub const SWAPON: libc::c_long = 167;
}

#[cfg(target_arch = "aarch64")]
mod syscall_nr {
    pub const FINIT_MODULE: libc::c_long = 273;
    pub const SWAPON: libc::c_long = 224;
}

// Mount flags
const MS_MOVE: libc::c_ulong = 0x2000;
const MS_NOATIME: libc::c_ulong = 0x400;

// Swap flags
const SWAP_FLAG_PREFER: libc::c_int = 0x8000;

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
macro_rules! log_fmt {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        unsafe {
            libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
            libc::write(2, b"\n".as_ptr() as *const libc::c_void, 1);
        }
    }};
}

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

fn load_module(path: &str, debug: bool) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);

    let path_cstr = match CString::new(path) {
        Ok(p) => p,
        Err(_) => {
            if debug {
                log_fmt!("[module] {}: invalid path", name);
            }
            return false;
        }
    };

    let fd = unsafe { libc::open(path_cstr.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        if debug {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
            log_fmt!("[module] {}: open failed (errno={})", name, errno);
        }
        return false;
    }

    let params = CString::new("").unwrap();
    let ret = unsafe {
        libc::syscall(
            syscall_nr::FINIT_MODULE,
            fd,
            params.as_ptr(),
            0 as libc::c_int,
        )
    };

    unsafe { libc::close(fd) };

    if ret == 0 {
        if debug {
            log_fmt!("[module] {}: ok", name);
        }
        true
    } else {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
        if errno == 17 {
            if debug {
                log_fmt!("[module] {}: built-in", name);
            }
            true
        } else {
            if debug {
                log_fmt!("[module] {}: errno={}", name, errno);
            }
            false
        }
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
        if let Ok(mut entries) = fs::read_dir("/sys/class/virtio-ports")
            && entries.any(|e| e.is_ok())
        {
            return true;
        }
        thread::sleep(Duration::from_micros(delay_us));
    }
    false
}

fn setup_zram(kver: &str) {
    log_fmt!("[zram] setup starting (kernel {})", kver);

    let m = format!("/lib/modules/{}/kernel", kver);

    // Load modules with logging
    let lz4_compress_ok = load_module(&format!("{}/lib/lz4/lz4_compress.ko", m), true);
    load_module(&format!("{}/crypto/lz4.ko", m), true);
    let zram_ok = load_module(&format!("{}/drivers/block/zram/zram.ko", m), true);

    if !zram_ok && !lz4_compress_ok {
        log_fmt!("[zram] module load failed, aborting");
        return;
    }

    // Wait for zram device to appear after module load
    // Exponential backoff: 1+2+4+8+16+32+64+128 = 255ms max
    // CI runners with nested virtualization may need longer waits
    log_fmt!("[zram] waiting for /sys/block/zram0...");
    let wait_times = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000];
    let mut total_wait_ms = 0u64;
    for delay_us in wait_times {
        if Path::new("/sys/block/zram0").exists() {
            log_fmt!("[zram] device appeared after {}ms", total_wait_ms);
            break;
        }
        thread::sleep(Duration::from_micros(delay_us));
        total_wait_ms += delay_us / 1000;
    }

    if !Path::new("/sys/block/zram0").exists() {
        log_fmt!(
            "[zram] device not found after {}ms, aborting",
            total_wait_ms
        );
        return;
    }

    // Use proper fallback chain: lz4 -> lzo-rle -> lzo
    let algorithms = ["lz4", "lzo-rle", "lzo"];
    let mut algo_set = false;
    for algo in algorithms {
        if fs::write("/sys/block/zram0/comp_algorithm", algo).is_ok() {
            log_fmt!("[zram] compression: {}", algo);
            algo_set = true;
            break;
        }
    }
    if !algo_set {
        log_fmt!("[zram] failed to set compression algorithm, aborting");
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
        log_fmt!("[zram] failed to read MemTotal, aborting");
        return;
    }

    // ZRAM_SIZE=$((MEM_KB * 512))
    let zram_size = mem_kb * 512;
    if fs::write("/sys/block/zram0/disksize", zram_size.to_string()).is_err() {
        log_fmt!("[zram] failed to set disksize, aborting");
        return;
    }
    log_fmt!("[zram] disksize: {} bytes (mem: {}KB)", zram_size, mem_kb);

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

    if let Err(e) = header_result {
        log_fmt!("[zram] mkswap failed: {}, aborting", e);
        return;
    }

    // swapon -p 100 /dev/zram0
    let dev = CString::new("/dev/zram0").unwrap();
    let ret = unsafe { libc::syscall(syscall_nr::SWAPON, dev.as_ptr(), SWAP_FLAG_PREFER | 100) };
    if ret < 0 {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
        log_fmt!("[zram] swapon failed (errno={}), aborting", errno);
        return;
    }

    log_fmt!("[zram] swap enabled");

    // VM tuning (these can fail silently - non-critical)
    let _ = fs::write("/proc/sys/vm/page-cluster", "0");
    let _ = fs::write("/proc/sys/vm/swappiness", "180");
    let _ = fs::write(
        "/proc/sys/vm/min_free_kbytes",
        (mem_kb * 4 / 100).to_string(),
    );
    let _ = fs::write("/proc/sys/vm/overcommit_memory", "0");

    log_fmt!("[zram] setup complete");
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
    log_fmt!("[init] ERROR: {}", msg);
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
    // SAFETY: called at startup before any threads are spawned
    unsafe {
        std::env::set_var("PATH", "/usr/local/bin:/usr/bin:/bin");
        std::env::set_var("HOME", "/root");
    }

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

    // execv only returns on error - report errno for debugging
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
    // Common errno: 2=ENOENT, 8=ENOEXEC (wrong arch), 13=EACCES, 14=EFAULT
    log_fmt!("[init] execv failed: errno={}", errno);

    error("execv guest-agent failed");
    fallback_shell();
}

fn main() {
    // Mount virtual filesystems
    mount("devtmpfs", "/dev", "devtmpfs", 0, "");
    // Shared memory for POSIX semaphores (Python multiprocessing, etc.)
    let _ = fs::create_dir("/dev/shm");
    mount("tmpfs", "/dev/shm", "tmpfs", 0, "size=64M,mode=1777");
    mount("proc", "/proc", "proc", 0, "");
    mount("sysfs", "/sys", "sysfs", 0, "");
    mount("tmpfs", "/tmp", "tmpfs", 0, "size=128M");

    // /dev/fd symlinks — not created by devtmpfs, must be done in userspace.
    // Required for bash process substitution <(), and /dev/std* for portability.
    // See: https://gitlab.alpinelinux.org/alpine/aports/-/issues/1465
    let _ = fs::remove_dir_all("/dev/fd"); // guard: devtmpfs may create it as a dir
    let _ = symlink("/proc/self/fd", "/dev/fd");
    let _ = symlink("/proc/self/fd/0", "/dev/stdin");
    let _ = symlink("/proc/self/fd/1", "/dev/stdout");
    let _ = symlink("/proc/self/fd/2", "/dev/stderr");

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
    load_module(&format!("{}/drivers/virtio/virtio_mmio.ko", m), false);
    load_module(&format!("{}/drivers/block/virtio_blk.ko", m), false);

    // Network modules (only if init.net=1)
    if need_net {
        load_module(&format!("{}/net/core/failover.ko", m), false);
        load_module(&format!("{}/drivers/net/net_failover.ko", m), false);
        load_module(&format!("{}/drivers/net/virtio_net.ko", m), false);
    }

    // Balloon (only if init.balloon=1)
    if need_balloon {
        load_module(&format!("{}/drivers/virtio/virtio_balloon.ko", m), false);
    }

    // Filesystem modules (always needed)
    load_module(&format!("{}/lib/crc16.ko", m), false);
    load_module(&format!("{}/crypto/crc32c_generic.ko", m), false);
    load_module(&format!("{}/lib/libcrc32c.ko", m), false);
    load_module(&format!("{}/fs/mbcache.ko", m), false);
    load_module(&format!("{}/fs/jbd2/jbd2.ko", m), false);
    load_module(&format!("{}/fs/ext4/ext4.ko", m), false);

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

//! Minimal init for QEMU microVMs
//!
//! Handles kernel modules, zram, devtmpfs, read-only rootfs mount, and switch_root.

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
const MS_NOSUID: libc::c_ulong = 0x2;
const MS_NODEV: libc::c_ulong = 0x4;
const MS_NOEXEC: libc::c_ulong = 0x8;
const MS_RDONLY: libc::c_ulong = 0x1;
const MS_NOATIME: libc::c_ulong = 0x400;
const MS_MOVE: libc::c_ulong = 0x2000;

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

    // Security: remove device node after swapon (kernel holds internal reference).
    remove_device_node("/dev/zram0");

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

fn remove_device_node(path: &str) {
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

    // Security: remove dangerous device nodes auto-created by devtmpfs.
    // /dev/mem exposes raw physical memory, /dev/kmem kernel virtual memory,
    // /dev/port raw I/O ports (x86). Even with CONFIG_STRICT_DEVMEM, the nodes
    // exist and leak kernel config info. Removing them is defense-in-depth.
    //
    // Why not lockdown=confidentiality? Alpine's linux-virt APKBUILD strips
    // modules (INSTALL_MOD_STRIP=1) which destroys the .PKCS7_message ELF
    // section containing signatures. Lockdown enforces signature verification,
    // so finit_module() returns EPERM for every module, preventing boot.
    // See: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=941827
    for path in ["/dev/mem", "/dev/kmem", "/dev/port"] {
        remove_device_node(path);
    }

    // Shared memory for POSIX semaphores (Python multiprocessing, etc.)
    // nosuid|nodev|noexec: CIS Benchmark 1.1.15 hardening. Won't break
    // multiprocessing — POSIX semaphores use shm_open()+mmap(), not execve().
    // Why 64MB: sufficient for POSIX semaphores and small shared memory segments.
    // Python multiprocessing default segment is ~1MB; larger allocations (numpy
    // shared arrays, torch tensors) need more but are out of scope for lightweight
    // sandboxes. 64MB is 25% of default 256MB guest RAM — a generous ceiling.
    let _ = fs::create_dir("/dev/shm");
    mount(
        "tmpfs",
        "/dev/shm",
        "tmpfs",
        MS_NOSUID | MS_NODEV | MS_NOEXEC,
        "size=64M,mode=1777,noswap",
    );
    // hidepid=2: hide /proc/[pid] entries for processes not owned by the
    // querying user. Prevents UID 1000 from reading /proc/1/maps (memory layout
    // leak). /proc/self is always exempted by the kernel. System-wide files
    // (/proc/meminfo, /proc/cpuinfo, /proc/net/*) are unaffected.
    // MS_MOVE in switch_root() preserves this option (superblock not modified).
    // Available since Linux 3.3; see proc(5).
    mount("proc", "/proc", "proc", 0, "hidepid=2");
    mount("sysfs", "/sys", "sysfs", 0, "");
    // Security: cgroupfs is intentionally NOT mounted.
    // Resource limits (memory, CPU, PIDs) are enforced on the host via cgroup v2
    // on the QEMU process (see src/exec_sandbox/cgroup.py). The guest runs a
    // single process (guest-agent), so internal cgroup subdivision is unnecessary.
    //
    // Not mounting cgroupfs eliminates cgroup-based escape vectors:
    // - CVE-2022-0492: write release_agent for root code execution
    // - CVE-2024-21626: leaked fd to /sys/fs/cgroup enables host filesystem access
    // /proc/cgroups will show registered subsystems (compiled-in), but they are
    // inert without a mounted cgroupfs (see cgroups(7)).

    // nosuid|nodev: CIS Benchmark 1.1.3–1.1.4 hardening.
    // noexec intentionally omitted: breaks uv wheel unpacking (pypa/pip#6364),
    // pnpm (pnpm#9776), PyInstaller, and user temp executables.
    // Why 128MB: half of default 256MB guest RAM. Balances scratch space for
    // pip/uv wheel builds (which unpack into /tmp) vs leaving RAM for user code.
    // nr_inodes=16384: fixed cap independent of VM memory (default is
    // totalram_pages/2, which varies 13K–55K+ depending on memory_mb). 16K
    // covers typical `pip install` (measured ~2K-5K files for large packages
    // like pandas).
    mount(
        "tmpfs",
        "/tmp",
        "tmpfs",
        MS_NOSUID | MS_NODEV,
        "size=128M,nr_inodes=16384,mode=1777,noswap",
    );

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

    // Mount root filesystem.
    // Default: read-only (ro,noatime,nosuid,nodev) — matches AWS Lambda rootfs flags.
    // init.rw=1: read-write — used for snapshot creation (package install to ext4).
    let need_rw = cmdline_has("init.rw=1");
    let mount_flags = if need_rw {
        MS_NOATIME | MS_NOSUID | MS_NODEV
    } else {
        MS_RDONLY | MS_NOATIME | MS_NOSUID | MS_NODEV
    };
    if mount("/dev/vda", "/mnt", "ext4", mount_flags, "") != 0 {
        // Fallback: try without specifying fstype
        if mount("/dev/vda", "/mnt", "", mount_flags, "") != 0 {
            error("mount /dev/vda failed");
            fallback_shell();
        }
    }

    // Security: remove block device node after mount. The kernel references the
    // device internally via bdevfs (indexed by major:minor), not the /dev path.
    // Prevents raw disk reads that bypass filesystem permissions.
    remove_device_node("/dev/vda");

    // Security: remove /dev/block/ directory (devtmpfs-created symlinks like 253:0 -> ../vda).
    // Eliminates alternative path to block device nodes.
    let _ = fs::remove_dir_all("/dev/block");

    // Security: harden kernel attack surface via sysctl.
    // These settings disable kernel subsystems that have been repeatedly exploited
    // for privilege escalation and sandbox escape. Each write may fail silently
    // if the sysctl doesn't exist (older kernel), which is acceptable.
    //
    // eBPF: CVE-2020-8835, CVE-2021-3490, CVE-2021-31440, CVE-2023-2163
    // Value 2 = disabled for all (even CAP_SYS_ADMIN requires BPF token)
    let _ = fs::write("/proc/sys/kernel/unprivileged_bpf_disabled", "2");
    //
    // io_uring: CVE-2023-2598, CVE-2024-0582
    // Value 2 = disabled for all users (requires kernel >= 6.6, introduced in 6.6-rc1)
    let _ = fs::write("/proc/sys/kernel/io_uring_disabled", "2");
    //
    // User namespaces: CVE-2022-0185, CVE-2023-0386
    // Value 0 = no unprivileged user namespaces (prevents gaining CAP_SYS_ADMIN)
    // unprivileged_userns_clone is Debian/Ubuntu-specific
    let _ = fs::write("/proc/sys/kernel/unprivileged_userns_clone", "0");
    // user.max_user_namespaces is the portable alternative (works on Alpine)
    let _ = fs::write("/proc/sys/user/max_user_namespaces", "0");
    //
    // Kernel address exposure: aids exploitation of CVE-2023-3269 (StackRot) etc.
    // Value 2 = restrict to CAP_SYSLOG (even root can't read without capability)
    let _ = fs::write("/proc/sys/kernel/kptr_restrict", "2");
    //
    // dmesg: kernel log may leak addresses, module info, hardware details
    // Value 1 = restrict to CAP_SYSLOG
    let _ = fs::write("/proc/sys/kernel/dmesg_restrict", "1");
    //
    // Perf events: can be used for side-channel attacks and kernel exploitation
    // Value 3 = disabled for all (even CAP_PERFMON)
    let _ = fs::write("/proc/sys/kernel/perf_event_paranoid", "3");
    //
    // BPF JIT hardening: prevents JIT spraying attacks (CVE-2024-56615)
    // Value 2 = JIT hardening for all users (blinding constants in BPF JIT)
    let _ = fs::write("/proc/sys/net/core/bpf_jit_harden", "2");
    //
    // userfaultfd: used in kernel race condition exploits to pause execution
    // at precise points during memory operations (UAF, double-free, TOCTOU).
    // Value 0 = restrict to CAP_SYS_PTRACE holders only.
    let _ = fs::write("/proc/sys/vm/unprivileged_userfaultfd", "0");
    //
    // YAMA ptrace_scope: restricts ptrace regardless of dumpable flag.
    // Value 2 = only CAP_SYS_PTRACE can ptrace (admin-only).
    // This is the primary ptrace defense — dumpable=0 is defense-in-depth
    // but exec() always resets dumpable to 1 (see begin_new_exec in fs/exec.c).
    // CVE-2022-30594: PTRACE_O_SUSPEND_SECCOMP bypass via ptrace.
    let _ = fs::write("/proc/sys/kernel/yama/ptrace_scope", "2");
    //
    // Filesystem link protections: prevent symlink/hardlink attacks in
    // world-writable directories (defense-in-depth for /tmp).
    let _ = fs::write("/proc/sys/fs/protected_symlinks", "1");
    let _ = fs::write("/proc/sys/fs/protected_hardlinks", "1");
    let _ = fs::write("/proc/sys/fs/protected_fifos", "2");
    let _ = fs::write("/proc/sys/fs/protected_regular", "2");
    //
    // suid_dumpable: controls core dump behavior for setuid processes.
    // Value 0 = no core dumps for processes that crossed privilege boundary.
    // Note: does NOT prevent same-UID exec from setting dumpable=1.
    let _ = fs::write("/proc/sys/fs/suid_dumpable", "0");
    //
    // SysRq: disable keyboard-triggered Magic SysRq functions.
    // NOTE: This does NOT protect /proc/sysrq-trigger — the kernel's
    // write_sysrq_trigger() bypasses the sysrq_enabled bitmask (check_mask=false
    // in drivers/tty/sysrq.c). The procfs trigger is blocked by a read-only
    // bind-mount in guest-agent instead. This only disables Alt+SysRq+key combos.
    // Value 0 = all keyboard SysRq functions disabled.
    let _ = fs::write("/proc/sys/kernel/sysrq", "0");
    //
    // Disable kernel module loading. MUST be last sysctl — once set to 1,
    // modules can never be loaded again (irreversible). All modules (virtio,
    // ext4, zram, etc.) are loaded above before this point.
    let _ = fs::write("/proc/sys/kernel/modules_disabled", "1");

    // No existence check - execv will fail if guest-agent missing
    switch_root();
}

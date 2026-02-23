//! Minimal init for QEMU microVMs
//!
//! Handles kernel modules, zram, devtmpfs, read-only rootfs mount, and switch_root.

#[macro_use]
mod sys;
mod device;
mod zram;

use std::ffi::CString;
use std::fs;
use std::os::unix::fs::symlink;

// Module paths relative to /lib/modules/<kver>/kernel/
const CORE_MODULES: &[&str] = &[
    "drivers/virtio/virtio_mmio.ko",
    "drivers/block/virtio_blk.ko",
];

const NET_MODULES: &[&str] = &[
    "net/core/failover.ko",
    "drivers/net/net_failover.ko",
    "drivers/net/virtio_net.ko",
];

const BALLOON_MODULES: &[&str] = &["drivers/virtio/virtio_balloon.ko"];

// Alpine 3.23 / kernel 6.18 paths
const FS_MODULES: &[&str] = &[
    "lib/crc/crc16.ko",
    "crypto/crc32c-cryptoapi.ko",
    "fs/mbcache.ko",
    "fs/jbd2/jbd2.ko",
    "fs/ext4/ext4.ko",
];

// Security: harden kernel attack surface via sysctl.
// These settings disable kernel subsystems that have been repeatedly exploited
// for privilege escalation and sandbox escape. Each write may fail silently
// if the sysctl doesn't exist (older kernel), which is acceptable.
const SYSCTL_HARDENING: &[(&str, &str)] = &[
    // eBPF: CVE-2020-8835, CVE-2021-3490, CVE-2021-31440, CVE-2023-2163
    // Value 2 = disabled for all (even CAP_SYS_ADMIN requires BPF token)
    ("/proc/sys/kernel/unprivileged_bpf_disabled", "2"),
    // User namespaces: CVE-2022-0185, CVE-2023-0386
    // Value 0 = no unprivileged user namespaces (prevents gaining CAP_SYS_ADMIN)
    // unprivileged_userns_clone is Debian/Ubuntu-specific
    ("/proc/sys/kernel/unprivileged_userns_clone", "0"),
    // user.max_user_namespaces is the portable alternative (works on Alpine)
    ("/proc/sys/user/max_user_namespaces", "0"),
    // Kernel address exposure: aids exploitation of CVE-2023-3269 (StackRot) etc.
    // Value 2 = restrict to CAP_SYSLOG (even root can't read without capability)
    ("/proc/sys/kernel/kptr_restrict", "2"),
    // dmesg: kernel log may leak addresses, module info, hardware details
    // Value 1 = restrict to CAP_SYSLOG
    ("/proc/sys/kernel/dmesg_restrict", "1"),
    // Perf events: can be used for side-channel attacks and kernel exploitation
    // Value 3 = disabled for all (even CAP_PERFMON)
    ("/proc/sys/kernel/perf_event_paranoid", "3"),
    // BPF JIT hardening: prevents JIT spraying attacks (CVE-2024-56615)
    // Value 2 = JIT hardening for all users (blinding constants in BPF JIT)
    ("/proc/sys/net/core/bpf_jit_harden", "2"),
    // userfaultfd: used in kernel race condition exploits to pause execution
    // at precise points during memory operations (UAF, double-free, TOCTOU).
    // Value 0 = restrict to CAP_SYS_PTRACE holders only.
    ("/proc/sys/vm/unprivileged_userfaultfd", "0"),
    // YAMA ptrace_scope: restricts ptrace regardless of dumpable flag.
    // Value 2 = only CAP_SYS_PTRACE can ptrace (admin-only).
    // CVE-2022-30594: PTRACE_O_SUSPEND_SECCOMP bypass via ptrace.
    ("/proc/sys/kernel/yama/ptrace_scope", "2"),
    // Filesystem link protections: prevent symlink/hardlink attacks in
    // world-writable directories (defense-in-depth for /tmp).
    ("/proc/sys/fs/protected_symlinks", "1"),
    ("/proc/sys/fs/protected_hardlinks", "1"),
    ("/proc/sys/fs/protected_fifos", "2"),
    ("/proc/sys/fs/protected_regular", "2"),
    // suid_dumpable: controls core dump behavior for setuid processes.
    // Value 0 = no core dumps for processes that crossed privilege boundary.
    // Note: does NOT prevent same-UID exec from setting dumpable=1.
    ("/proc/sys/fs/suid_dumpable", "0"),
    // SysRq: disable keyboard-triggered Magic SysRq functions.
    // NOTE: This does NOT protect /proc/sysrq-trigger — the kernel's
    // write_sysrq_trigger() bypasses the sysrq_enabled bitmask (check_mask=false
    // in drivers/tty/sysrq.c). The procfs trigger is blocked by a read-only
    // bind-mount in guest-agent instead. This only disables Alt+SysRq+key combos.
    // Value 0 = all keyboard SysRq functions disabled.
    ("/proc/sys/kernel/sysrq", "0"),
    // Thread bomb mitigation: limit total system-wide threads.
    // Default is memory-proportional (~1,659 on a 512MB VM, computed as
    // mempages / (8 * THREAD_SIZE / PAGE_SIZE) in kernel/fork.c:fork_init).
    // Value 1200 leaves headroom for ~150 kernel threads (kthreadd, kworker,
    // ksoftirqd, etc.) while capping user-space thread bombs. RLIMIT_NPROC=1024
    // per-UID is the primary defense; this is defense-in-depth.
    // See: https://docs.kernel.org/admin-guide/sysctl/kernel.html
    ("/proc/sys/kernel/threads-max", "1200"),
    // MUST be last — disable kernel module loading (irreversible once set to 1).
    // All modules (virtio, ext4, zram, etc.) are loaded above before this point.
    ("/proc/sys/kernel/modules_disabled", "1"),
];

fn apply_sysctl_hardening() {
    for &(path, value) in SYSCTL_HARDENING {
        let _ = fs::write(path, value);
    }
}

fn load_modules(kver: &str) {
    let m = format!("/lib/modules/{}/kernel", kver);
    let load = |modules: &[&str]| {
        for module in modules {
            sys::load_module(&format!("{}/{}", m, module), false);
        }
    };
    load(CORE_MODULES);
    if sys::cmdline_has("init.net=1") {
        load(NET_MODULES);
    }
    if sys::cmdline_has("init.balloon=1") {
        load(BALLOON_MODULES);
    }
    load(FS_MODULES);
}

fn mount_virtual_filesystems() {
    sys::mount("devtmpfs", "/dev", "devtmpfs", 0, "");

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
        device::remove_device_node(path);
    }

    // Shared memory for POSIX semaphores (Python multiprocessing, etc.)
    // nosuid|nodev|noexec: CIS Benchmark 1.1.15 hardening. Won't break
    // multiprocessing — POSIX semaphores use shm_open()+mmap(), not execve().
    // Why 64MB: sufficient for POSIX semaphores and small shared memory segments.
    // Python multiprocessing default segment is ~1MB; larger allocations (numpy
    // shared arrays, torch tensors) need more but are out of scope for lightweight
    // sandboxes. 64MB is 25% of default 256MB guest RAM — a generous ceiling.
    let _ = fs::create_dir("/dev/shm");
    sys::mount(
        "tmpfs",
        "/dev/shm",
        "tmpfs",
        sys::MS_NOSUID | sys::MS_NODEV | sys::MS_NOEXEC,
        "size=64M,mode=1777,noswap",
    );

    // hidepid=2: hide /proc/[pid] entries for processes not owned by the
    // querying user. Prevents UID 1000 from reading /proc/1/maps (memory layout
    // leak). /proc/self is always exempted by the kernel. System-wide files
    // (/proc/meminfo, /proc/cpuinfo, /proc/net/*) are unaffected.
    // MS_MOVE in switch_root() preserves this option (superblock not modified).
    // Available since Linux 3.3; see proc(5).
    sys::mount("proc", "/proc", "proc", 0, "hidepid=2");
    sys::mount("sysfs", "/sys", "sysfs", 0, "");

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
    sys::mount(
        "tmpfs",
        "/tmp",
        "tmpfs",
        sys::MS_NOSUID | sys::MS_NODEV,
        "size=128M,nr_inodes=16384,mode=1777,noswap",
    );
}

fn setup_dev_symlinks() {
    // /dev/fd symlinks — not created by devtmpfs, must be done in userspace.
    // Required for bash process substitution <(), and /dev/std* for portability.
    // See: https://gitlab.alpinelinux.org/alpine/aports/-/issues/1465
    let _ = fs::remove_dir_all("/dev/fd"); // guard: devtmpfs may create it as a dir
    let _ = symlink("/proc/self/fd", "/dev/fd");
    let _ = symlink("/proc/self/fd/0", "/dev/stdin");
    let _ = symlink("/proc/self/fd/1", "/dev/stdout");
    let _ = symlink("/proc/self/fd/2", "/dev/stderr");
}

fn mount_rootfs() {
    // Mount root filesystem.
    // Default: read-only (ro,noatime,nosuid,nodev) — matches AWS Lambda rootfs flags.
    // init.rw=1: read-write — used for snapshot creation (package install to ext4).
    let need_rw = sys::cmdline_has("init.rw=1");
    let mount_flags = if need_rw {
        sys::MS_NOATIME | sys::MS_NOSUID | sys::MS_NODEV
    } else {
        sys::MS_RDONLY | sys::MS_NOATIME | sys::MS_NOSUID | sys::MS_NODEV
    };
    if sys::mount("/dev/vda", "/mnt", "ext4", mount_flags, "") != 0 {
        // Fallback: try without specifying fstype
        if sys::mount("/dev/vda", "/mnt", "", mount_flags, "") != 0 {
            sys::error("mount /dev/vda failed");
            sys::fallback_shell();
        }
    }

    // Security: remove block device node after mount. The kernel references the
    // device internally via bdevfs (indexed by major:minor), not the /dev path.
    // Prevents raw disk reads that bypass filesystem permissions.
    device::remove_device_node("/dev/vda");

    // Security: remove /dev/block/ directory (devtmpfs-created symlinks like 253:0 -> ../vda).
    // Eliminates alternative path to block device nodes.
    let _ = fs::remove_dir_all("/dev/block");
}

fn switch_root() -> ! {
    // cd /mnt
    if std::env::set_current_dir("/mnt").is_err() {
        sys::error("chdir /mnt failed");
        sys::fallback_shell();
    }

    // mount --move /dev dev etc
    sys::mount_move("/dev", "dev");
    sys::mount_move("/proc", "proc");
    sys::mount_move("/sys", "sys");
    sys::mount_move("/tmp", "tmp");

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
            sys::MS_MOVE,
            std::ptr::null(),
        );
    }

    // Step 2: chroot .
    let chroot_ret = unsafe { libc::chroot(dot.as_ptr()) };
    if chroot_ret != 0 {
        sys::error("chroot failed");
        sys::fallback_shell();
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

    // Stage 1: Ensure stdin is valid (open /dev/null if needed)
    let devnull = CString::new("/dev/null").unwrap();
    let stdin_fd = unsafe { libc::open(devnull.as_ptr(), libc::O_RDONLY) };
    if stdin_fd >= 0 && stdin_fd != 0 {
        unsafe {
            libc::dup2(stdin_fd, 0);
            libc::close(stdin_fd);
        }
    }

    // Stage 2: Verify stdout/stderr are valid, fix if needed
    let stdout_valid = unsafe { libc::fcntl(1, libc::F_GETFD) } >= 0;
    let stderr_valid = unsafe { libc::fcntl(2, libc::F_GETFD) } >= 0;

    if !stdout_valid || !stderr_valid {
        let mut fds_buf = [0i32; 2];
        let mut count = 0;
        if !stdout_valid {
            fds_buf[count] = 1;
            count += 1;
        }
        if !stderr_valid {
            fds_buf[count] = 2;
            count += 1;
        }
        device::open_console(
            device::CONSOLE_DEVICES_WITH_FALLBACK,
            libc::O_WRONLY,
            &fds_buf[..count],
        );
    }

    // Stage 3: Redirect console for guest-agent
    // Order: hvc0 (virtio-console), ttyS0 (ISA serial), ttyAMA0 (ARM PL011 UART)
    // vm_manager uses hvc0 for x86 and ttyAMA0 for ARM64
    device::open_console(device::CONSOLE_DEVICES, libc::O_RDWR, &[0, 1, 2]);

    // exec /usr/local/bin/guest-agent
    let prog = CString::new("/usr/local/bin/guest-agent").unwrap();
    let args: [*const libc::c_char; 2] = [prog.as_ptr(), std::ptr::null()];
    unsafe { libc::execv(prog.as_ptr(), args.as_ptr()) };

    // execv only returns on error - report errno for debugging
    let errno = sys::last_errno();
    // Common errno: 2=ENOENT, 8=ENOEXEC (wrong arch), 13=EACCES, 14=EFAULT
    log_fmt!("[init] execv failed: errno={}", errno);

    sys::error("execv guest-agent failed");
    sys::fallback_shell();
}

fn main() {
    mount_virtual_filesystems();
    setup_dev_symlinks();
    device::redirect_to_console();

    let kver = match sys::get_kernel_version() {
        Some(v) => v,
        None => {
            sys::error("uname failed");
            sys::fallback_shell();
        }
    };

    load_modules(&kver);
    zram::setup_zram(&kver);

    if !device::wait_for_block_device("/dev/vda") {
        sys::error("timeout waiting for /dev/vda");
    }
    device::wait_for_virtio_ports();
    device::setup_virtio_ports();

    mount_rootfs();
    apply_sysctl_hardening();
    switch_root();
}

#[cfg(test)]
mod tests {
    use super::*;

    // SYSCTL_HARDENING data invariant tests

    #[test]
    fn sysctl_modules_disabled_is_last() {
        // CRITICAL: modules_disabled=1 is irreversible, must be last
        let last = SYSCTL_HARDENING.last().unwrap();
        assert_eq!(last.0, "/proc/sys/kernel/modules_disabled");
        assert_eq!(last.1, "1");
    }

    #[test]
    fn sysctl_no_duplicate_paths() {
        let mut seen = std::collections::HashSet::new();
        for &(path, _) in SYSCTL_HARDENING {
            assert!(seen.insert(path), "duplicate sysctl: {}", path);
        }
    }

    #[test]
    fn sysctl_paths_are_valid() {
        for &(path, value) in SYSCTL_HARDENING {
            assert!(
                path.starts_with("/proc/sys/"),
                "invalid sysctl path: {}",
                path
            );
            assert!(!value.is_empty(), "empty value for: {}", path);
        }
    }

    #[test]
    fn sysctl_table_not_empty() {
        assert!(
            SYSCTL_HARDENING.len() >= 15,
            "sysctl table unexpectedly small"
        );
    }

    #[test]
    fn sysctl_values_are_numeric() {
        // All current sysctl values are integers
        for &(path, value) in SYSCTL_HARDENING {
            assert!(
                value.parse::<u32>().is_ok(),
                "non-numeric value '{}' for: {}",
                value,
                path
            );
        }
    }

    // Module loading array data invariant tests

    #[test]
    fn module_paths_end_with_ko() {
        for &path in CORE_MODULES
            .iter()
            .chain(NET_MODULES)
            .chain(BALLOON_MODULES)
            .chain(FS_MODULES)
        {
            assert!(path.ends_with(".ko"), "not a .ko file: {}", path);
        }
    }

    #[test]
    fn module_paths_are_relative() {
        for &path in CORE_MODULES
            .iter()
            .chain(NET_MODULES)
            .chain(BALLOON_MODULES)
            .chain(FS_MODULES)
        {
            assert!(!path.starts_with('/'), "should be relative: {}", path);
        }
    }

    #[test]
    fn module_no_duplicates() {
        let all: Vec<&str> = CORE_MODULES
            .iter()
            .chain(NET_MODULES)
            .chain(BALLOON_MODULES)
            .chain(FS_MODULES)
            .copied()
            .collect();
        let mut seen = std::collections::HashSet::new();
        for path in &all {
            assert!(seen.insert(path), "duplicate module: {}", path);
        }
    }

    #[test]
    fn ext4_module_present() {
        assert!(FS_MODULES.contains(&"fs/ext4/ext4.ko"));
    }
}

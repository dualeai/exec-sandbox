//! Minimal init for QEMU microVMs
//!
//! Custom kernel (CONFIG_MODULES=n): all drivers are built-in.
//! Handles devtmpfs, read-only rootfs mount, and switch_root.

#[macro_use]
mod sys;
mod device;
mod zram;

use std::ffi::CString;
use std::fs;

fn mount_virtual_filesystems() {
    sys::mount("devtmpfs", "/dev", "devtmpfs", 0, "");

    // Security: remove dangerous device nodes auto-created by devtmpfs.
    // /dev/mem exposes raw physical memory, /dev/kmem kernel virtual memory,
    // /dev/port raw I/O ports (x86). Even with CONFIG_STRICT_DEVMEM, the nodes
    // exist and leak kernel config info. Removing them is defense-in-depth.
    for path in ["/dev/mem", "/dev/kmem", "/dev/port"] {
        device::remove_device_node(path);
    }

    // B4: /dev/shm mount deferred to guest-agent (only needed for Python multiprocessing).
    // Guest-agent mounts it lazily before first use, saving ~2-5ms on critical path.

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

// B5: /dev/fd symlinks deferred to guest-agent phase 2.
// Not needed for guest-agent startup or kernel boot. Required for bash
// process substitution <() and /dev/std* portability.
// See: https://gitlab.alpinelinux.org/alpine/aports/-/issues/1465

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
            log_error!("mount /dev/vda failed");
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
        log_error!("chdir /mnt failed");
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
        log_error!("chroot failed");
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

    // execv only returns on error
    // Common errno: 2=ENOENT, 8=ENOEXEC (wrong arch), 13=EACCES, 14=EFAULT
    log_error!(
        "execv /usr/local/bin/guest-agent failed: errno={}",
        sys::last_errno()
    );
    sys::fallback_shell();
}

fn main() {
    let t0 = sys::monotonic_us();
    mount_virtual_filesystems();
    let t1 = sys::monotonic_us();

    // B5: /dev symlinks deferred to guest-agent

    device::redirect_to_console();
    log_info!(
        "[timing] mount_vfs: {}.{}ms",
        (t1 - t0) / 1000,
        ((t1 - t0) % 1000) / 100
    );

    // B2: Wait for vda + virtio-ports in parallel (independent devices)
    let vda_handle = std::thread::spawn(|| device::wait_for_block_device("/dev/vda"));
    let virtio_ok = device::wait_for_virtio_ports();
    let vda_ok = vda_handle.join().unwrap_or(false);
    if !vda_ok {
        log_error!("timeout waiting for /dev/vda");
    }
    if virtio_ok {
        device::setup_virtio_ports();
    }
    let t2 = sys::monotonic_us();
    log_info!(
        "[timing] wait_devices: {}.{}ms",
        (t2 - t1) / 1000,
        ((t2 - t1) % 1000) / 100
    );

    mount_rootfs();
    let t3 = sys::monotonic_us();
    log_info!(
        "[timing] mount_rootfs: {}.{}ms",
        (t3 - t2) / 1000,
        ((t3 - t2) % 1000) / 100
    );

    log_info!(
        "[timing] init_total: {}.{}ms",
        (t3 - t0) / 1000,
        ((t3 - t0) % 1000) / 100
    );
    switch_root();
}

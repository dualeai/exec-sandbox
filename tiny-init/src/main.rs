//! Minimal init for QEMU microVMs
//!
//! Custom kernel (CONFIG_MODULES=n): all drivers are built-in.
//! Mounts devtmpfs, /proc, /sys, /tmp and the rootfs (EROFS, with an optional
//! ext4 snapshot layer), moves into the new root (busybox switch_root steps),
//! formats and mounts the home disk on /home/user, then execs the guest agent.

#[macro_use]
mod sys;
mod device;

use std::ffi::CString;
use std::fs;
use std::os::unix::process::CommandExt;

fn mount_virtual_filesystems() {
    sys::mount("devtmpfs", "/dev", "devtmpfs", 0, "");

    // Security: remove dangerous device nodes auto-created by devtmpfs.
    // /dev/mem exposes raw physical memory, /dev/kmem kernel virtual memory,
    // /dev/port raw I/O ports (x86). Even with CONFIG_STRICT_DEVMEM, the nodes
    // exist and leak kernel config info. Removing them is defense-in-depth.
    for path in ["/dev/mem", "/dev/kmem", "/dev/port"] {
        device::remove_device_node(path);
    }

    // Security: restrict /dev/urandom and /dev/random to owner-write (0644).
    // devtmpfs creates these as 0666, allowing any user to inject entropy via
    // write(). The RO bind-mount of /dev (guest-agent) does NOT block char
    // device writes (Linux VFS exempts S_ISCHR from sb_permission RO check).
    // Guest-agent (root) needs write for RNDRESEEDCRNG ioctl; user code
    // (UID 1000) only needs read or getrandom() syscall.
    device::chmod_paths(0o644, &["/dev/urandom", "/dev/random"]);

    // The guest agent mounts /dev/shm in its phase-2 setup (setup_dev_shm).
    // Only Python multiprocessing and similar code need it.

    // hidepid=2: hide /proc/[pid] entries for processes not owned by the
    // querying user. Prevents UID 1000 from reading /proc/1/maps (memory layout
    // leak). /proc/self is always exempted by the kernel. System-wide files
    // (/proc/meminfo, /proc/cpuinfo, /proc/net/*) are unaffected.
    // MS_MOVE in enter_rootfs() preserves this option (superblock not modified).
    // Available since Linux 3.3; see proc(5).
    sys::mount("proc", "/proc", "proc", 0, "hidepid=2");
    sys::mount("sysfs", "/sys", "sysfs", 0, "");

    // Security: cgroupfs is intentionally NOT mounted.
    // Resource limits (memory, CPU, PIDs) are enforced on the host via cgroup v2
    // on the QEMU process (see src/exec_sandbox/cgroup.py). The guest runs one
    // workload (the guest agent and its children), so it needs no cgroup split.
    //
    // Not mounting cgroupfs eliminates cgroup-based escape vectors:
    // - CVE-2022-0492: write release_agent for root code execution
    // - CVE-2024-21626: leaked fd to /sys/fs/cgroup enables host filesystem access
    // /proc/cgroups will show registered subsystems (compiled-in), but they are
    // inert without a mounted cgroupfs (see cgroups(7)).

    // nosuid|nodev: CIS Benchmark 1.1.3–1.1.4 hardening.
    // noexec intentionally omitted: breaks uv wheel unpacking (pypa/pip#6364),
    // pnpm (pnpm#9776), PyInstaller, and user temp executables.
    // Size = 50% of RAM: scales with VM memory instead of fixed cap.
    // Per-UID quota (usrquota + usrquota_block_hardlimit) prevents sparse file
    // inflation attacks — must be set at initial mount time, not remount.
    // nr_inodes=16384: fixed cap independent of VM memory (default is
    // totalram_pages/2, which varies 13K–55K+ depending on memory_mb). 16K
    // covers typical `pip install` (measured ~2K-5K files for large packages
    // like pandas).
    // Page-align (round down) so the mount size is an exact multiple of PAGE_SIZE.
    // Without this, the kernel rounds up non-aligned values, causing statvfs
    // to report a larger size than the computed half-memory value.
    let page_mask = !(sys::page_size() - 1);
    let half_mem_bytes = (sys::read_mem_total_kb() / 2 * 1024) & page_mask;
    sys::mount(
        "tmpfs",
        "/tmp",
        "tmpfs",
        libc::MS_NOSUID | libc::MS_NODEV,
        &format!(
            "size={half_mem_bytes},nr_inodes=16384,mode=1777,usrquota,usrquota_block_hardlimit={half_mem_bytes},noswap"
        ),
    );
}

// The /dev/fd and /dev/std* symlinks: the guest agent makes them in phase 2.
// Not needed for guest-agent startup or kernel boot. Required for bash
// process substitution <() and /dev/std* portability.
// See: https://gitlab.alpinelinux.org/alpine/aports/-/issues/1465

/// Find a block device by its virtio serial number.
/// Reads /sys/block/vd*/serial and returns the /dev/vdX path matching `target_serial`.
/// ARM virt (MMIO) enumerates virtio devices in reverse declaration order,
/// so /dev/vda is NOT always the first drive declared in QEMU args.
fn find_block_device_by_serial(target_serial: &str) -> Option<String> {
    let entries = match fs::read_dir("/sys/block") {
        Ok(e) => e,
        Err(_) => return None,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("vd") {
            continue;
        }
        let serial_path = entry.path().join("serial");
        if let Ok(serial) = fs::read_to_string(&serial_path)
            && serial.trim() == target_serial
        {
            return Some(format!("/dev/{}", name_str));
        }
    }
    None
}

/// Find a drive the host attached, retrying briefly (enumeration race on a
/// loaded host): 6 checks, 10 ms apart. Built-in virtio-blk drivers probe
/// before init, so a miss here is rare; a drive that stays missing is fatal
/// (fail closed).
fn require_block_device_by_serial(serial: &str) -> String {
    let mut dev = None;
    device::poll_backoff(&[10_000; 5], || {
        dev = find_block_device_by_serial(serial);
        dev.is_some()
    });
    let Some(dev) = dev else {
        log_error!("drive serial={} not found", serial);
        sys::fallback_shell();
    };
    dev
}

/// Format the blank snapshot drive `dev` as ext4 (no journal, no reserved
/// blocks) with mkfs.ext4 from the EROFS rootfs mounted at /lower.
///
/// The tool's ELF interpreter resolves against the initramfs root (where /lib
/// is empty), so bind-mount /dev into the rootfs and chroot into it first.
fn mkfs_in_lower(dev: &str) -> bool {
    let root_dev = "/lower/dev";
    if sys::mount("/dev", root_dev, "", libc::MS_BIND, "") != 0 {
        log_error!(
            "bind mount /dev -> {} failed: errno={}",
            root_dev,
            sys::last_errno()
        );
        return false;
    }
    let root_c = CString::new("/lower").unwrap();
    let status = unsafe {
        std::process::Command::new("/sbin/mkfs.ext4")
            .args(["-q", "-O", "^has_journal", "-m", "0", dev])
            .pre_exec(move || {
                let slash = CString::new("/").unwrap();
                if libc::chroot(root_c.as_ptr()) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::chdir(slash.as_ptr()) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            })
            .status()
    };
    sys::umount(root_dev);
    matches!(status, Ok(s) if s.success())
}

fn mount_rootfs() {
    // Discover drives by serial number (set in QEMU -device virtio-blk serial=).
    // "base" = EROFS rootfs, "snap" = ext4 snapshot overlay.
    // Cannot rely on /dev/vda ordering: ARM MMIO enumerates in reverse.
    let base_dev = find_block_device_by_serial("base").unwrap_or_else(|| "/dev/vda".to_string());
    let snap_dev = sys::cmdline_has("init.snap=1").then(|| require_block_device_by_serial("snap"));

    let erofs_flags = libc::MS_RDONLY | libc::MS_NOATIME | libc::MS_NOSUID | libc::MS_NODEV;
    let need_rw = sys::cmdline_has("init.rw=1");

    if let Some(ref snap) = snap_dev {
        // Two-drive mode: EROFS base + ext4 overlay, merged via overlayfs.
        // Used for:
        //   - Snapshot creation (init.rw=1): ext4 writable upper
        //   - Snapshot usage: ext4 read-only, overlayfs provides merged view
        //
        // Mount hierarchy (on initramfs, before enter_rootfs):
        //   /lower      ← EROFS base (always ro)
        //   /upper_dev  ← ext4 snapshot (rw or ro)
        //   /mnt        ← overlayfs merge (enter_rootfs target)

        let _ = fs::create_dir("/lower");
        let _ = fs::create_dir("/upper_dev");

        // Mount EROFS base as lower layer
        if sys::mount(&base_dev, "/lower", "erofs", erofs_flags, "") != 0 {
            log_error!("mount {} (erofs) failed", base_dev);
            sys::fallback_shell();
        }

        // Mount ext4 overlay layer (rw for snapshot creation, ro for snapshot usage)
        if need_rw {
            // Snapshot creation: format the blank drive (EROFS rootfs already at /lower).
            if !mkfs_in_lower(snap) {
                log_error!("mkfs.ext4 {} failed", snap);
                sys::fallback_shell();
            }
        }
        let snap_flags = if need_rw {
            libc::MS_NOATIME | libc::MS_NOSUID | libc::MS_NODEV
        } else {
            libc::MS_RDONLY | libc::MS_NOATIME | libc::MS_NOSUID | libc::MS_NODEV
        };
        if sys::mount(snap, "/upper_dev", "ext4", snap_flags, "") != 0 {
            log_error!("mount {} (ext4) failed", snap);
            sys::fallback_shell();
        }

        if need_rw {
            // Snapshot creation: overlayfs with writable upper
            let _ = fs::create_dir_all("/upper_dev/upper");
            let _ = fs::create_dir_all("/upper_dev/work");
            let ovl_opts = "lowerdir=/lower,upperdir=/upper_dev/upper,workdir=/upper_dev/work";
            if sys::mount("overlay", "/mnt", "overlay", libc::MS_NOATIME, ovl_opts) != 0 {
                log_error!("overlayfs mount failed (rw)");
                sys::fallback_shell();
            }
        } else {
            // Snapshot usage: overlayfs with two read-only lower layers.
            // No upper/work needed for read-only overlay.
            let ovl_opts = "lowerdir=/upper_dev/upper:/lower";
            if sys::mount(
                "overlay",
                "/mnt",
                "overlay",
                libc::MS_RDONLY | libc::MS_NOATIME,
                ovl_opts,
            ) != 0
            {
                log_error!("overlayfs mount failed (ro)");
                sys::fallback_shell();
            }
        }
    } else {
        // Single-drive mode: EROFS only, no snapshot overlay.
        if sys::mount(&base_dev, "/mnt", "erofs", erofs_flags, "") != 0 {
            log_error!("mount {} (erofs) failed", base_dev);
            sys::fallback_shell();
        }
    }

    // Security: remove block device nodes after mount. The kernel references the
    // device internally via bdevfs (indexed by major:minor), not the /dev path.
    // Prevents raw disk reads that bypass filesystem permissions.
    device::remove_device_node(&base_dev);
    if let Some(ref snap) = snap_dev {
        device::remove_device_node(snap);
    }

    // Security: remove /dev/block/ directory (devtmpfs-created symlinks like 253:0 -> ../vda).
    // Eliminates alternative path to block device nodes.
    let _ = fs::remove_dir_all("/dev/block");
}

/// Format the VM's home disk (serial=home) and mount it on /home/user.
///
/// Every VM has one: a raw sparse file the host attaches. tiny-init runs only
/// at a cold boot (a restore or a load resumes saved RAM), so the disk is
/// always blank here. Runs inside the new root (after enter_rootfs): mkfs runs
/// from the rootfs directly, with no /dev bind mount, whose umount would wait a
/// full RCU grace period (~12 ms at HZ=100 with rcu_normal_after_boot).
///
/// mkfs.ext4 options:
/// - assume_storage_prezeroed: the sparse file reads as zeros, so mke2fs skips
///   zeroing and ext4lazyinit never runs (~60 ms at any size).
/// - root_owner: /home/user belongs to the sandbox user.
/// - nodiscard: a blank disk has nothing to discard.
/// - ^orphan_file: the mount reads 5 blocks, not 69 (the classic orphan list is
///   journaled too; it only scales worse for many CPUs).
/// - ^resize_inode: nothing grows a home filesystem online.
/// - stdin and stdout on /dev/null: mke2fs probes for an old filesystem (libblkid)
///   only on a tty. stderr stays on the console for errors.
fn mount_home() {
    let home = require_block_device_by_serial("home");
    let status = std::process::Command::new("/sbin/mkfs.ext4")
        .args([
            "-q",
            "-F",
            "-b",
            "4096",
            "-m",
            "0",
            "-O",
            "^orphan_file,^resize_inode",
            "-E",
            "assume_storage_prezeroed=1,root_owner=1000:1000,nodiscard",
            &home,
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status();
    if !matches!(status, Ok(s) if s.success()) {
        log_error!("mkfs.ext4 {} failed", home);
        sys::fallback_shell();
    }
    let flags = libc::MS_NOATIME | libc::MS_NOSUID | libc::MS_NODEV;
    if sys::mount(&home, "/home/user", "ext4", flags, "discard") != 0 {
        log_error!(
            "mount {} on /home/user failed: errno={}",
            home,
            sys::last_errno()
        );
        sys::fallback_shell();
    }
    // mke2fs makes lost+found as root 0700: uid 1000 could not read it, so
    // tar, du or rm over /home/user would fail. Nothing runs fsck on this disk.
    if let Err(e) = std::fs::remove_dir("/home/user/lost+found") {
        log_warn!("remove /home/user/lost+found failed: {}", e);
    }
    // Same rule as the root drives: no raw device node after mount.
    device::remove_device_node(&home);
}

/// Make the merged rootfs at /mnt the root: /dev, /proc, /sys and /tmp move into it.
fn enter_rootfs() {
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
    // 4. exec new init (exec_agent, after mount_home)
    // See: https://docs.kernel.org/filesystems/ramfs-rootfs-initramfs.html

    let dot = CString::new(".").unwrap();
    let root = CString::new("/").unwrap();

    // Step 1: mount --move . / (overmount rootfs)
    unsafe {
        libc::mount(
            dot.as_ptr(),
            root.as_ptr(),
            std::ptr::null(),
            libc::MS_MOVE,
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
}

/// Set up the environment and stdio, then exec the guest agent.
fn exec_agent() -> ! {
    // Set minimal environment for guest-agent
    // SAFETY: called at startup before any threads are spawned
    unsafe {
        std::env::set_var("PATH", "/usr/local/bin:/usr/bin:/bin");
        std::env::set_var("HOME", "/root");
    }

    // Stage 1: point stdin at /dev/null (stage 3 replaces it when a console opens)
    let devnull = CString::new("/dev/null").unwrap();
    let stdin_fd = unsafe { libc::open(devnull.as_ptr(), libc::O_RDONLY) };
    if stdin_fd > 0 {
        unsafe {
            libc::dup2(stdin_fd, 0);
            libc::close(stdin_fd);
        }
    }

    // Stage 2: Verify stdout/stderr are valid, fix if needed
    let invalid: Vec<libc::c_int> = [1, 2]
        .into_iter()
        .filter(|&fd| unsafe { libc::fcntl(fd, libc::F_GETFD) } < 0)
        .collect();
    if !invalid.is_empty() {
        device::open_console(
            device::CONSOLE_DEVICES_WITH_FALLBACK,
            libc::O_WRONLY,
            &invalid,
        );
    }

    // Stage 3: point stdin, stdout and stderr at the first console that opens.
    // Order: hvc0 (virtio-console), ttyS0 (ISA serial), ttyAMA0 (ARM PL011 UART).
    // The host (qemu_cmd.py) sets console=hvc0 on x86_64, console=ttyAMA0 on aarch64.
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

/// Log the time of one init step, from `from_us` to `to_us` (monotonic, in µs).
fn log_timing(name: &str, from_us: u64, to_us: u64) {
    let us = to_us - from_us;
    log_info!("[timing] {}: {}.{}ms", name, us / 1000, (us % 1000) / 100);
}

fn main() {
    let t0 = sys::monotonic_us();
    mount_virtual_filesystems();
    let t1 = sys::monotonic_us();

    device::redirect_to_console();
    log_timing("mount_vfs", t0, t1);

    // Wait for vda and the virtio ports in parallel (independent devices)
    // /dev/vda always exists (first virtio-blk enumerated by kernel), though its
    // actual role (base vs snapshot) varies by architecture — mount_rootfs resolves
    // the correct mapping via serial numbers.
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
    log_timing("wait_devices", t1, t2);

    mount_rootfs();
    let t3 = sys::monotonic_us();
    log_timing("mount_rootfs", t2, t3);
    // The mount_home time also counts enter_rootfs (five mount moves and a chroot).
    enter_rootfs();
    mount_home();
    let t4 = sys::monotonic_us();
    log_timing("mount_home", t3, t4);

    log_timing("init_total", t0, t4);
    exec_agent();
}

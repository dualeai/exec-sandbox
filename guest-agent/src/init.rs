//! Init-time setup: PID 1 environment, mounts, network, zombie reaping.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::SetOnce;

use crate::constants::NETWORK_SETUP_TIMEOUT_SECONDS;

/// Network readiness gate: the outcome of network setup, set once within
/// NETWORK_SETUP_TIMEOUT_SECONDS of boot (Ok also for an intentionally offline
/// VM). ExecuteCode, InstallPackages and WarmRepl wait on it; Ping and file I/O
/// respond immediately.
static NETWORK_STATE: SetOnce<Result<(), &'static str>> = SetOnce::const_new();

/// Capped-zram safety gate: true once zram swap is active with its pressure
/// guard and /dev is sealed. Untrusted process spawning is blocked until then.
/// Production has no intentional no-swap mode: every setup failure makes the VM
/// unavailable. main.rs awaits zram setup before the virtio listener starts, so
/// the gate is final when a command reads it.
static ZRAM_READY: AtomicBool = AtomicBool::new(false);

// ============================================================================
// Zombie reaping
// ============================================================================

/// The children that agent code waits for (REPLs, package managers): one entry
/// per spawn, by registration number, with the PID and, once the PID 1 reaper
/// collected that child, its raw wait status. waitpid(-1) in the reaper can
/// collect such a child before tokio does. Its PID can then come back as
/// another process (an orphan is a child of PID 1 too), which waitpid on that
/// PID would find: a later child with the same PID gets its own entry.
static WAITED_CHILDREN: std::sync::Mutex<BTreeMap<u64, (u32, Option<libc::c_int>)>> =
    std::sync::Mutex::new(BTreeMap::new());

static NEXT_WAITED_CHILD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn waited_children() -> std::sync::MutexGuard<'static, BTreeMap<u64, (u32, Option<libc::c_int>)>> {
    // A panic while held cannot leave the map half written: use it as it is.
    WAITED_CHILDREN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn register_waited(pid: u32) -> u64 {
    let key = NEXT_WAITED_CHILD.fetch_add(1, Ordering::Relaxed);
    waited_children().insert(key, (pid, None));
    key
}

/// Keep the status that the reaper collected for the newest entry of `pid`
/// (an older entry of that PID is a child that ended before).
fn keep_waited_exit(pid: u32, status: libc::c_int) {
    let mut children = waited_children();
    if let Some((_, slot)) = children.values_mut().rev().find(|(p, _)| *p == pid)
        && slot.is_none()
    {
        *slot = Some(status);
    }
}

fn reaped_status(key: u64) -> Option<std::process::ExitStatus> {
    use std::os::unix::process::ExitStatusExt;
    waited_children()
        .get(&key)
        .and_then(|(_, status)| *status)
        .map(std::process::ExitStatus::from_raw)
}

/// pidfd_send_signal flag: the signal goes to the process group of the pidfd's
/// process (<linux/pidfd.h>, Linux 6.9).
const PIDFD_SIGNAL_PROCESS_GROUP: libc::c_uint = 1 << 2;

/// A child that agent code waits for, with its entry in WAITED_CHILDREN.
///
/// Every wait reads the status that the reaper kept first, and no signal goes
/// to a child that the reaper collected: its PID may belong to another process.
/// The process group that the child leads gets signals through a pidfd: once
/// the child is collected too (its members can outlive it), and never another
/// group with the same ID.
pub(crate) struct WaitedChild {
    child: tokio::process::Child,
    /// Key in WAITED_CHILDREN.
    key: u64,
    /// A pidfd of the child, opened before any wait: the kernel keeps its group
    /// while a member lives, whatever process gets the PID later.
    pidfd: std::os::fd::OwnedFd,
}

impl WaitedChild {
    /// Register right after the spawn, with no await between: the runtime has
    /// one thread, so the reaper cannot collect the child before it is known.
    /// A child whose pidfd does not open (a file limit) gets SIGKILL, and the
    /// error comes back.
    pub(crate) fn register(mut child: tokio::process::Child) -> std::io::Result<Self> {
        use std::os::fd::FromRawFd;
        // A child that no wait polled yet has its PID
        let pid = child.id().expect("a child just spawned has its PID");
        // SAFETY: a plain syscall; the result is checked before use.
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0) };
        if fd < 0 {
            let error = std::io::Error::last_os_error();
            let _ = child.start_kill();
            return Err(error);
        }
        // SAFETY: `fd` is a new descriptor that nothing else owns.
        let pidfd = unsafe { std::os::fd::OwnedFd::from_raw_fd(fd as libc::c_int) };
        Ok(Self {
            child,
            key: register_waited(pid),
            pidfd,
        })
    }

    fn reaped(&self) -> Option<std::process::ExitStatus> {
        reaped_status(self.key)
    }

    /// Keep the status that tokio collected in the entry: keep_waited_exit then
    /// skips it when the PID comes back as another process that PID 1 reaps.
    fn collected(&self, status: std::process::ExitStatus) -> std::process::ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        if let Some((_, slot)) = waited_children().get_mut(&self.key)
            && slot.is_none()
        {
            *slot = Some(status.into_raw());
        }
        status
    }

    /// Send `signal` to the process group that the child leads (ESRCH: no
    /// member left), also once the child was collected.
    pub(crate) fn signal_group(&self, signal: libc::c_int) -> std::io::Result<()> {
        use std::os::fd::AsRawFd;
        // SAFETY: a valid pidfd, and no siginfo.
        let sent = unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                self.pidfd.as_raw_fd(),
                signal,
                std::ptr::null::<libc::siginfo_t>(),
                PIDFD_SIGNAL_PROCESS_GROUP,
            )
        };
        if sent < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// The PID while the child can get a signal: None once tokio or the PID 1
    /// reaper collected it.
    pub(crate) fn id(&self) -> Option<u32> {
        match self.reaped() {
            Some(_) => None,
            None => self.child.id(),
        }
    }

    /// The exit status if the child ended, else None.
    pub(crate) fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        match self.reaped() {
            Some(status) => Ok(Some(status)),
            None => Ok(self.child.try_wait()?.map(|status| self.collected(status))),
        }
    }

    /// Wait for the child to end.
    pub(crate) async fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        if let Some(status) = self.reaped() {
            return Ok(status);
        }
        match self.child.wait().await {
            // The reaper collected it during the wait.
            Err(e) if e.raw_os_error() == Some(libc::ECHILD) => self.reaped().ok_or(e),
            waited => waited.map(|status| self.collected(status)),
        }
    }

    /// Send SIGKILL, unless the child was collected.
    pub(crate) fn start_kill(&mut self) -> std::io::Result<()> {
        match self.reaped() {
            Some(_) => Ok(()),
            None => self.child.start_kill(),
        }
    }
}

impl Drop for WaitedChild {
    fn drop(&mut self) {
        waited_children().remove(&self.key);
    }
}

/// Reap zombie processes when running as PID 1. The wait status of a
/// registered child goes into its WAITED_CHILDREN entry.
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
            let mut status: libc::c_int = 0;
            let pid = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG) };
            if pid <= 0 {
                break;
            }
            keep_waited_exit(pid as u32, status);
        }
    }
}

// ============================================================================
// Init environment setup
// ============================================================================

/// Phase 1 — minimal setup for Ping response (fast, <1ms).
/// Sets environment variables only. Called before listen_virtio_serial().
pub(crate) fn setup_phase1() {
    // SAFETY: called at startup before any threads are spawned
    unsafe { std::env::set_var("PATH", "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin") };
    unsafe { std::env::set_var("UV_NO_CACHE", "1") };
}

/// Phase 2 core — sync init: chmod, mounts, sysctl hardening, dev setup (~10ms).
/// Called from spawn_blocking before the ports open. After it, network setup
/// runs in the background, and main.rs awaits zram setup before it opens the ports.
pub(crate) fn setup_phase2_core() {
    // CIS Benchmark 6.1.x compliance (libc::chmod, no fork/exec)
    chmod_paths(0o755, &[c"/etc", c"/usr", c"/var", c"/sbin", c"/bin"]);
    chmod_paths(
        0o644,
        &[
            c"/etc/passwd",
            c"/etc/group",
            c"/etc/resolv.conf",
            c"/etc/hosts",
        ],
    );
    chmod_paths(0o640, &[c"/etc/shadow"]);

    // /home/user: ext4 home disk, mounted by tiny-init before the agent starts.
    // ALL /proc/sys writes MUST happen before mount_readonly_paths() which
    // makes /proc/sys read-only. This includes sysctl hardening AND zram
    // VM tuning (swappiness, page-cluster, etc.).
    apply_sysctls();
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
    // /dev writes (symlinks, shm mount) happen here, before /dev becomes
    // read-only in setup_zram_and_seal_dev().
    setup_dev_symlinks();
    setup_dev_shm();
    mount_readonly_paths();
}

/// Zram setup + /dev sealing, run to completion before the virtio listener
/// opens its long-lived ports (main.rs awaits this). No guest-side deadline:
/// the blocking worker cannot be cancelled, and abandoning it would leave an
/// armed pressure guard and live swap inside a VM already published as failed.
/// If the worker hangs, the host boot timeout discards the VM. There is no
/// no-swap path.
pub(crate) async fn setup_zram_and_seal_dev() {
    let zram_active = match tokio::task::spawn_blocking(setup_zram_swap).await {
        Ok(true) => true,
        Ok(false) => {
            log_warn!("[zram] setup failed; marking VM unavailable");
            false
        }
        Err(error) => {
            log_warn!("[zram] setup worker failed: {error}");
            false
        }
    };
    let dev_read_only = mount_readonly(c"/dev", false); // its nodes are the guest's devices
    if !dev_read_only {
        log_warn!("[zram] failed to seal /dev read-only; marking VM unavailable");
    }
    ZRAM_READY.store(zram_active && dev_read_only, Ordering::Release);
}

/// Background network setup: interface config + gvproxy verification.
/// Runs on spawn_blocking (blocking netlink and probe I/O). The blocking
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
        Ok(Ok(Ok(()))) => Ok(()),
        Ok(Ok(Err(error))) => {
            log_warn!("network setup failed: {error}");
            Err("network setup failed")
        }
        Ok(Err(error)) => {
            log_warn!("network setup worker failed: {error}");
            Err("network setup failed")
        }
        Err(_) => {
            log_warn!(
                "network setup exceeded {}s; marking VM unavailable",
                NETWORK_SETUP_TIMEOUT_SECONDS
            );
            Err("network setup failed")
        }
    };
    // The only setter: it runs once, at boot.
    let _ = NETWORK_STATE.set(state);
    let t_done = crate::monotonic_ms();
    if state.is_ok() {
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

/// Process-spawning commands require both boot workers' outcomes. Zram is
/// final before the listener starts (main.rs awaits it), so it is checked
/// first and at once: anything but active swap fails closed, and `swapon`
/// cannot overlap untrusted process execution before monitor startup. Only
/// network setup is still concurrent (setup_network_background sets its
/// outcome within its deadline). Intentionally offline VMs are ready after
/// loopback setup; a failed or stuck worker is an infrastructure error so the
/// host can destroy and replace the VM.
pub(crate) async fn wait_for_untrusted_readiness() -> Result<(), &'static str> {
    if !ZRAM_READY.load(Ordering::Acquire) {
        return Err("zram safety setup failed");
    }
    *NETWORK_STATE.wait().await
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
            let _ = mount_readonly(path, true);
        }
    }

    // No log line per mount: each serial write is slow.
    // /dev is sealed by setup_zram_and_seal_dev() only after zram has removed its
    // setup-only block-device node. Untrusted execution remains gated until that
    // read-only remount succeeds.
    // No /proc/sysrq-trigger to seal: the kernel has no MAGIC_SYSRQ.
    for path in [c"/etc/hosts", c"/etc/resolv.conf", c"/proc/sys"] {
        let _ = mount_readonly(path, true);
    }

    // Mask sensitive /proc files (bind-mount /dev/null → reads return empty).
    // Safe to mask here: rw_mode and QUIET_MODE already read /proc/cmdline above.
    // No /proc/keys to mask: the kernel has no KEYS (no keyring).
    for path in [
        // Boot params: rootfstype, init flags, console config (reconnaissance)
        c"/proc/cmdline",
        // Exact kernel version string (aids CVE matching against guest kernel)
        c"/proc/version",
        // Interrupt counters per CPU — thermal side-channel attack vector
        // (Docker masked this in Mar 2025, GHSA-6fw5-f8r9-fgfm)
        c"/proc/interrupts",
        // High-resolution timer internals — timing side-channel
        // (already EPERM via dmesg_restrict, mask as defense-in-depth)
        c"/proc/timer_list",
    ] {
        mount_mask(path);
    }
}

/// Configure loopback and eth0 network interfaces.
/// No wait loop for eth0 — configure if present, skip if not.
/// Called from spawn_blocking in setup_network_background().
///
/// The agent sends the rtnetlink requests of `ip` itself: each `ip` command
/// would fork and run a binary from the rootfs (3 ms on HVF, 10 ms on KVM), on
/// the path of the first exec.
fn setup_network() -> Result<(), String> {
    let rtnl = Rtnetlink::open().map_err(|e| format!("stage=netlink_socket error={e}"))?;
    network_stage("loopback_up", || rtnl.link_up(c"lo"))?;

    if std::path::Path::new("/sys/class/net/eth0").exists() {
        network_stage("eth0_up", || rtnl.link_up(c"eth0"))?;
        network_stage("eth0_address", || {
            rtnl.add_address(c"eth0", [192, 168, 127, 2], 24)
        })?;
        network_stage("default_route", || {
            rtnl.add_default_route([192, 168, 127, 1])
        })?;

        // Verify gvproxy connectivity. ExecuteCode, InstallPackages and
        // WarmRepl gate on the untrusted-readiness check (network + zram), so
        // a failure here makes each of them fail rather than run without networking.
        verify_gvproxy().map_err(str::to_string)?;
    } else {
        log_info!("eth0 not found, intentionally offline network is ready");
    }
    Ok(())
}

fn network_stage(stage: &str, step: impl FnOnce() -> std::io::Result<()>) -> Result<(), String> {
    let started = crate::monotonic_ms();
    log_info!("[network] stage={stage} action=start at={started}ms");
    match step() {
        Ok(()) => {
            log_info!(
                "[network] stage={stage} action=complete elapsed={}ms",
                crate::monotonic_ms() - started
            );
            Ok(())
        }
        Err(error) => Err(format!(
            "stage={stage} error={error} elapsed={}ms",
            crate::monotonic_ms() - started
        )),
    }
}

/// A NETLINK_ROUTE socket: one request at a time, each acknowledged by the
/// kernel. The requests are those that `ip` (iproute2) sends.
struct Rtnetlink {
    fd: std::os::fd::OwnedFd,
    seq: std::cell::Cell<u32>,
}

impl Rtnetlink {
    const HEADER_LEN: usize = 16; // struct nlmsghdr

    fn open() -> std::io::Result<Self> {
        use std::os::fd::FromRawFd;
        // SAFETY: a plain syscall; the result is checked before use.
        let fd = unsafe {
            libc::socket(
                libc::AF_NETLINK,
                libc::SOCK_RAW | libc::SOCK_CLOEXEC,
                libc::NETLINK_ROUTE,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            // SAFETY: `fd` is a new descriptor that nothing else owns.
            fd: unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) },
            seq: std::cell::Cell::new(0),
        })
    }

    /// `ip link set <name> up`.
    fn link_up(&self, name: &std::ffi::CStr) -> std::io::Result<()> {
        let body = link_up_body(interface_index(name)?);
        self.request(libc::RTM_NEWLINK, 0, &body)
    }

    /// `ip addr add <address>/<prefix_len> dev <name>` (no broadcast address).
    fn add_address(
        &self,
        name: &std::ffi::CStr,
        address: [u8; 4],
        prefix_len: u8,
    ) -> std::io::Result<()> {
        let body = address_body(interface_index(name)?, address, prefix_len);
        self.request(libc::RTM_NEWADDR, Self::CREATE_EXCL, &body)
    }

    /// `ip route add default via <gateway>`.
    fn add_default_route(&self, gateway: [u8; 4]) -> std::io::Result<()> {
        self.request(
            libc::RTM_NEWROUTE,
            Self::CREATE_EXCL,
            &default_route_body(gateway),
        )
    }

    const CREATE_EXCL: u16 = (libc::NLM_F_CREATE | libc::NLM_F_EXCL) as u16;

    /// Send one request and wait for its acknowledgement: Ok, or the kernel's error.
    fn request(&self, kind: u16, flags: u16, body: &[u8]) -> std::io::Result<()> {
        use std::os::fd::AsRawFd;
        let seq = self.seq.get().wrapping_add(1);
        self.seq.set(seq);
        let len = Self::HEADER_LEN + body.len();
        let flags = (libc::NLM_F_REQUEST | libc::NLM_F_ACK) as u16 | flags;
        let mut message = Vec::with_capacity(len);
        message.extend_from_slice(&(len as u32).to_ne_bytes());
        message.extend_from_slice(&kind.to_ne_bytes());
        message.extend_from_slice(&flags.to_ne_bytes());
        message.extend_from_slice(&seq.to_ne_bytes());
        message.extend_from_slice(&0u32.to_ne_bytes()); // port id: the kernel sets it
        message.extend_from_slice(body);

        // SAFETY: sockaddr_nl is plain data; zero is its default for every field.
        let mut kernel: libc::sockaddr_nl = unsafe { std::mem::zeroed() };
        kernel.nl_family = libc::AF_NETLINK as libc::sa_family_t;
        // SAFETY: valid buffer and address, with their lengths.
        let sent = unsafe {
            libc::sendto(
                self.fd.as_raw_fd(),
                message.as_ptr().cast(),
                message.len(),
                0,
                (&kernel as *const libc::sockaddr_nl).cast(),
                std::mem::size_of::<libc::sockaddr_nl>() as libc::socklen_t,
            )
        };
        if sent < 0 {
            return Err(std::io::Error::last_os_error());
        }

        let mut reply = [0u8; 4096];
        loop {
            // SAFETY: `reply` is valid for writes of its full length.
            let n = unsafe {
                libc::recv(
                    self.fd.as_raw_fd(),
                    reply.as_mut_ptr().cast(),
                    reply.len(),
                    0,
                )
            };
            if n < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            if let Some(result) = netlink_ack(&reply[..n as usize], seq) {
                return result;
            }
        }
    }
}

/// The acknowledgement of request `seq` among the netlink messages of `reply`
/// (an NLMSG_ERROR message: error 0 is success), or None if it is not there.
fn netlink_ack(reply: &[u8], seq: u32) -> Option<std::io::Result<()>> {
    let field = |at: usize| -> [u8; 4] { reply[at..at + 4].try_into().unwrap() };
    let mut at = 0;
    while at + Rtnetlink::HEADER_LEN <= reply.len() {
        let len = u32::from_ne_bytes(field(at)) as usize;
        let kind = u16::from_ne_bytes([reply[at + 4], reply[at + 5]]);
        let message_seq = u32::from_ne_bytes(field(at + 8));
        if len < Rtnetlink::HEADER_LEN || at + len > reply.len() {
            return None;
        }
        if kind == libc::NLMSG_ERROR as u16
            && message_seq == seq
            && len >= Rtnetlink::HEADER_LEN + 4
        {
            let error = i32::from_ne_bytes(field(at + Rtnetlink::HEADER_LEN));
            return Some(if error == 0 {
                Ok(())
            } else {
                Err(std::io::Error::from_raw_os_error(-error))
            });
        }
        at += len.next_multiple_of(4);
    }
    None
}

/// struct ifinfomsg that sets IFF_UP: family, padding, type, index, flags, change.
fn link_up_body(index: i32) -> Vec<u8> {
    let up = libc::IFF_UP as u32;
    let mut body = vec![libc::AF_UNSPEC as u8, 0, 0, 0];
    body.extend_from_slice(&index.to_ne_bytes());
    body.extend_from_slice(&up.to_ne_bytes());
    body.extend_from_slice(&up.to_ne_bytes());
    body
}

/// struct ifaddrmsg (family, prefix length, flags, scope, index), then the
/// local address and the address, as `ip addr add` sends them with no peer.
fn address_body(index: i32, address: [u8; 4], prefix_len: u8) -> Vec<u8> {
    let mut body = vec![libc::AF_INET as u8, prefix_len, 0, libc::RT_SCOPE_UNIVERSE];
    body.extend_from_slice(&index.to_ne_bytes());
    push_route_attr(&mut body, libc::IFA_LOCAL, &address);
    push_route_attr(&mut body, libc::IFA_ADDRESS, &address);
    body
}

/// struct rtmsg (family, dst_len, src_len, tos, table, protocol, scope, type,
/// flags) of a default route in the main table, then its gateway.
fn default_route_body(gateway: [u8; 4]) -> Vec<u8> {
    let mut body = vec![
        libc::AF_INET as u8,
        0,
        0,
        0,
        libc::RT_TABLE_MAIN,
        libc::RTPROT_BOOT,
        libc::RT_SCOPE_UNIVERSE,
        libc::RTN_UNICAST,
    ];
    body.extend_from_slice(&0u32.to_ne_bytes());
    push_route_attr(&mut body, libc::RTA_GATEWAY, &gateway);
    body
}

/// Append a route attribute (struct rtattr and its data, padded to 4 bytes).
fn push_route_attr(body: &mut Vec<u8>, kind: u16, data: &[u8]) {
    body.extend_from_slice(&((4 + data.len()) as u16).to_ne_bytes());
    body.extend_from_slice(&kind.to_ne_bytes());
    body.extend_from_slice(data);
    body.resize(body.len().next_multiple_of(4), 0);
}

fn interface_index(name: &std::ffi::CStr) -> std::io::Result<i32> {
    // SAFETY: a NUL-terminated name.
    match unsafe { libc::if_nametoindex(name.as_ptr()) } {
        0 => Err(std::io::Error::last_os_error()),
        index => Ok(index as i32),
    }
}

// ============================================================================
// /dev setup done by the agent, not by tiny-init (off its critical path)
// ============================================================================

/// Create /dev/fd and /dev/std* symlinks.
/// Required for bash process substitution <() and /dev/std* portability.
fn setup_dev_symlinks() {
    use std::os::unix::fs::symlink;
    let _ = symlink("/proc/self/fd", "/dev/fd");
    let _ = symlink("/proc/self/fd/0", "/dev/stdin");
    let _ = symlink("/proc/self/fd/1", "/dev/stdout");
    let _ = symlink("/proc/self/fd/2", "/dev/stderr");
}

/// Mount /dev/shm for POSIX shared memory / semaphores.
/// Only needed for Python multiprocessing and similar use cases.
///
/// Sized to 40% of RAM, with a per-UID quota (`usrquota` +
/// `usrquota_block_hardlimit`): each UID may use up to the mount size, the same
/// bound as `size=`, so it adds no limit for the sandbox user. Quota options take
/// effect only at the initial mount, not at a remount.
fn setup_dev_shm() {
    // Page-align (round down) so the mount size is an exact multiple of PAGE_SIZE.
    let page_mask = !(page_size() - 1);
    // 40% of RAM
    let shm_bytes = (read_mem_total_kb() * 2048 / 5) & page_mask;
    let opts = format!("size={shm_bytes},usrquota,usrquota_block_hardlimit={shm_bytes},noswap");
    let data = std::ffi::CString::new(opts).unwrap();

    // devtmpfs makes no /dev/shm: the mount point comes first
    let _ = std::fs::create_dir("/dev/shm");
    // SAFETY: NUL-terminated strings that outlive the call.
    let mounted = unsafe {
        libc::mount(
            c"tmpfs".as_ptr(),
            c"/dev/shm".as_ptr(),
            c"tmpfs".as_ptr(),
            libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC,
            data.as_ptr() as *const libc::c_void,
        )
    };
    // Non-fatal: only needed for multiprocessing.
    if mounted != 0 {
        log_warn!("/dev/shm mount failed: {}", std::io::Error::last_os_error());
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

/// Set up the capped zram swap and its pressure guard: true once both are
/// active, false after a failure (cleaned up).
fn setup_zram_swap() -> bool {
    use std::io::Write;
    use std::path::Path;
    use std::thread;
    use std::time::Duration;

    const SWAP_FLAG_PREFER: libc::c_int = 0x8000; // <sys/swap.h>

    // Wait for the built-in zram device to appear.
    // Exponential backoff: 0.5+1+2+4+8+16 = 31.5ms max
    let zram_exists = || Path::new("/sys/block/zram0").exists();
    let mut found = zram_exists();
    for delay_us in [500, 1000, 2000, 4000, 8000, 16000] {
        if found {
            break;
        }
        thread::sleep(Duration::from_micros(delay_us));
        found = zram_exists();
    }
    if !found {
        log_warn!("[zram] setup failed: device not found");
        return false;
    }

    let requested_algorithm = "lz4";
    let dev = c"/dev/zram0";

    // Configure geometry and activate swap. Every step here is a reset-only
    // failure — swapon is the last fallible step, so any Err leaves swap
    // inactive and a bare zram reset is always the correct cleanup. The
    // closure boundary ends exactly at swapon; oom_guard.spawn() and remove_file
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
            if unsafe { libc::swapon(dev.as_ptr(), SWAP_FLAG_PREFER | 100) } < 0 {
                return Err(format!("swapon: {}", std::io::Error::last_os_error()));
            }
            Ok((oom_guard, geometry))
        })();
    let (oom_guard, geometry) = match setup {
        Ok(v) => v,
        Err(e) => {
            log_warn!("[zram] setup failed: {e}");
            let _ = std::fs::write("/sys/block/zram0/reset", "1");
            return false;
        }
    };

    // Start only after verified zram activation. If the dedicated thread cannot
    // be created, immediately remove swap and reset zram rather than leave the
    // capped geometry active without its pressure guard.
    if let Err(e) = oom_guard.spawn() {
        log_warn!("[zram] setup failed: could not start OOM guard: {e}");
        let swapoff_rc = unsafe { libc::swapoff(dev.as_ptr()) };
        let reset_result = std::fs::write("/sys/block/zram0/reset", "1");
        if swapoff_rc != 0 || reset_result.is_err() {
            log_warn!("[zram] cleanup after guard failure failed, powering off guest");
            crate::oom_guard::emergency_poweroff();
        }
        return false;
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
        return false;
    }

    // VM tuning for zram is pre-applied in apply_zram_vm_tuning() before
    // /proc/sys is made read-only. No sysctl writes needed here.
    true
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

/// Sysctl hardening, applied in phase 2 core (before the ports open, so before
/// any request). The kernel has no BPF syscall, no userfaultfd and no SysRq
/// (images/kernel/exec-sandbox.config): their sysctls do not exist. Perf events
/// stay on x86 only (its arch Kconfig selects them).
const SYSCTLS: &[(&str, &str)] = &[
    // User namespaces: CVE-2022-0185, CVE-2023-0386
    ("/proc/sys/user/max_user_namespaces", "0"),
    // Kernel address exposure
    ("/proc/sys/kernel/kptr_restrict", "2"),
    // dmesg restriction
    ("/proc/sys/kernel/dmesg_restrict", "1"),
    // Perf events
    ("/proc/sys/kernel/perf_event_paranoid", "3"),
    // BPF JIT hardening (cBPF: seccomp filters)
    ("/proc/sys/net/core/bpf_jit_harden", "2"),
    // YAMA ptrace_scope
    ("/proc/sys/kernel/yama/ptrace_scope", "2"),
    // Filesystem link protections
    ("/proc/sys/fs/protected_symlinks", "1"),
    ("/proc/sys/fs/protected_hardlinks", "1"),
    ("/proc/sys/fs/protected_fifos", "2"),
    ("/proc/sys/fs/protected_regular", "2"),
    // suid_dumpable
    ("/proc/sys/fs/suid_dumpable", "0"),
    // Thread bomb mitigation
    ("/proc/sys/kernel/threads-max", "1200"),
];

fn apply_sysctls() {
    for &(path, value) in SYSCTLS {
        let _ = std::fs::write(path, value);
    }
}

/// Pre-apply zram VM tuning sysctls before /proc/sys is made read-only.
/// These values are safe to set before zram device setup completes: they
/// apply at once, and the swap-related ones have no effect until swapon.
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
/// Runs inside the background network worker; ExecuteCode, InstallPackages and
/// WarmRepl wait on the untrusted-readiness gate, which reflects this probe's outcome.
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

/// Set file mode on multiple paths using libc::chmod (no fork/exec overhead).
fn chmod_paths(mode: libc::mode_t, paths: &[&std::ffi::CStr]) {
    for path in paths {
        unsafe { libc::chmod(path.as_ptr(), mode) };
    }
}

/// Mask a path by bind-mounting /dev/null over it (returns empty on read).
fn mount_mask(path: &std::ffi::CStr) {
    let ret = unsafe {
        libc::mount(
            c"/dev/null".as_ptr(),
            path.as_ptr(),
            std::ptr::null(),
            libc::MS_BIND,
            std::ptr::null(),
        )
    };
    if ret != 0 {
        log_warn!(
            "mask mount {} failed: {}",
            path.to_string_lossy(),
            std::io::Error::last_os_error()
        );
    }
}

/// Bind-mount a path onto itself and remount read-only with nosuid, and nodev when `nodev`.
/// A bind remount keeps only the flags it names, so it names each one (the overlay root has
/// neither). /dev takes no nodev: it would refuse every open of its device nodes.
fn mount_readonly(path: &std::ffi::CStr, nodev: bool) -> bool {
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
            libc::MS_BIND
                | libc::MS_REMOUNT
                | libc::MS_RDONLY
                | libc::MS_NOSUID
                | if nodev { libc::MS_NODEV } else { 0 },
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
    use crate::error::{
        exit_code_from_status, graceful_terminate_process_group, wait_child_bounded,
    };

    /// Collect `child` before tokio and keep its status, as the PID 1 reaper does.
    fn reap_first(child: &WaitedChild) {
        let pid = child.child.id().unwrap() as libc::pid_t;
        let mut status: libc::c_int = 0;
        // SAFETY: waits for this test's own child only (waitpid(-1) would take
        // the children of the tests that run in parallel).
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        keep_waited_exit(pid as u32, status);
    }

    #[tokio::test]
    async fn a_child_that_the_reaper_collected_first_keeps_its_exit_code() {
        let spawned = tokio::process::Command::new("sh")
            .args(["-c", "kill -9 $$"])
            .spawn()
            .unwrap();
        let mut child = WaitedChild::register(spawned).unwrap();
        reap_first(&child);

        // tokio alone loses the status: the REPL would look alive-then-gone.
        let lost = child.child.try_wait().unwrap_err();
        assert_eq!(lost.raw_os_error(), Some(libc::ECHILD));
        let status = child.try_wait().unwrap().unwrap();
        assert_eq!(exit_code_from_status(status), 137);
        let status = child.wait().await.unwrap();
        assert_eq!(exit_code_from_status(status), 137);
        assert_eq!(child.id(), None);
    }

    #[tokio::test]
    async fn a_collected_child_whose_pid_came_back_gets_no_wait_and_no_signal() {
        // The REPL ended (exit 3) and the reaper kept its status; another process
        // now has its PID. The live `sleep` plays that process: waitpid on the
        // PID finds it, and a signal to the PID would reach it. (The group
        // signals go through the pidfd: they never reach a later process.)
        let spawned = tokio::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
        let pid = spawned.id().unwrap();
        let mut child = WaitedChild::register(spawned).unwrap();
        keep_waited_exit(pid, 3 << 8);

        assert_eq!(child.try_wait().unwrap().and_then(|s| s.code()), Some(3));
        assert_eq!(child.id(), None);
        child.start_kill().unwrap();
        let status =
            wait_child_bounded(&mut child, "test", std::time::Duration::from_secs(2)).await;
        assert_eq!(status.and_then(|s| s.code()), Some(3));
        // The first fatal signal sets the exit status: SIGINT from here, unless the
        // agent signalled first (SIGKILL from start_kill).
        // SAFETY: the test's own child.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGINT) };
        let mut status: libc::c_int = 0;
        assert_eq!(
            unsafe { libc::waitpid(pid as libc::pid_t, &mut status, 0) },
            pid as libc::pid_t
        );
        assert_eq!(
            libc::WTERMSIG(status),
            libc::SIGINT,
            "the agent signalled the PID"
        );
    }

    #[tokio::test]
    async fn the_members_of_a_collected_leader_still_end() {
        // The REPL (sh) ended and the reaper collected it; its member holds the
        // stdout pipe, so the pipe ends only when the group SIGKILL reaches it.
        let mut spawned = tokio::process::Command::new("sh")
            .args(["-c", "sleep 30 & exit 0"])
            .stdout(std::process::Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap();
        let mut stdout = spawned.stdout.take().unwrap();
        let mut child = WaitedChild::register(spawned).unwrap();
        reap_first(&child);
        assert_eq!(child.id(), None);

        graceful_terminate_process_group(&mut child, 5).await;

        let mut rest = Vec::new();
        let ended = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            tokio::io::AsyncReadExt::read_to_end(&mut stdout, &mut rest),
        )
        .await;
        assert!(
            matches!(ended, Ok(Ok(0))),
            "a member of the group still holds the pipe: {ended:?}"
        );
    }

    /// Block until `pid` exits, without collecting it (WNOWAIT): tokio collects it next.
    fn wait_exit_uncollected(pid: u32) {
        // SAFETY: `info` is valid for writes; waits for this test's own child only.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let waited = unsafe {
            libc::waitid(
                libc::P_PID,
                pid as libc::id_t,
                &mut info,
                libc::WEXITED | libc::WNOWAIT,
            )
        };
        assert_eq!(waited, 0);
    }

    #[tokio::test]
    async fn a_child_that_tokio_collected_keeps_its_status_when_its_pid_comes_back() {
        // tokio collects each child first (try_wait, then wait); later the reaper
        // collects another process with the same PID (played by the call below).
        let spawn = || {
            tokio::process::Command::new("sh")
                .args(["-c", "exit 3"])
                .spawn()
                .unwrap()
        };
        let mut by_try_wait = WaitedChild::register(spawn()).unwrap();
        let pid = by_try_wait.child.id().unwrap();
        wait_exit_uncollected(pid);
        assert_eq!(
            by_try_wait.try_wait().unwrap().and_then(|s| s.code()),
            Some(3)
        );
        keep_waited_exit(pid, 7 << 8);
        assert_eq!(
            by_try_wait.try_wait().unwrap().and_then(|s| s.code()),
            Some(3)
        );

        let mut by_wait = WaitedChild::register(spawn()).unwrap();
        let pid = by_wait.child.id().unwrap();
        assert_eq!(by_wait.wait().await.unwrap().code(), Some(3));
        keep_waited_exit(pid, 7 << 8);
        assert_eq!(by_wait.wait().await.unwrap().code(), Some(3));
        assert_eq!(by_wait.id(), None);
    }

    #[test]
    fn a_new_child_with_the_pid_of_an_ended_one_keeps_its_own_status() {
        let pid = u32::MAX - 7; // no real process: only the map is used
        let first = register_waited(pid);
        keep_waited_exit(pid, 3 << 8); // the first child ended, unread
        let second = register_waited(pid); // the PID came back for a new child
        assert_eq!(reaped_status(first).and_then(|s| s.code()), Some(3));
        assert_eq!(reaped_status(second), None);

        keep_waited_exit(pid, 5 << 8);
        assert_eq!(reaped_status(second).and_then(|s| s.code()), Some(5));
        assert_eq!(reaped_status(first).and_then(|s| s.code()), Some(3));
        waited_children().remove(&first);
        assert_eq!(reaped_status(second).and_then(|s| s.code()), Some(5));
        waited_children().remove(&second);
    }

    #[test]
    fn the_reaper_keeps_no_status_of_a_child_that_nobody_waits_for() {
        let pid = u32::MAX - 8;
        let key = register_waited(pid);
        waited_children().remove(&key); // the registration dropped
        keep_waited_exit(pid, 3 << 8);
        assert!(waited_children().values().all(|(p, _)| *p != pid));
    }

    // -------------------------------------------------------------------------
    // rtnetlink requests of the network setup
    // -------------------------------------------------------------------------

    fn ne16(v: u16) -> [u8; 2] {
        v.to_ne_bytes()
    }

    #[test]
    fn link_up_body_is_an_ifinfomsg_that_sets_iff_up() {
        let mut expected = vec![0u8, 0, 0, 0]; // AF_UNSPEC, padding, type
        expected.extend_from_slice(&7i32.to_ne_bytes());
        expected.extend_from_slice(&1u32.to_ne_bytes()); // IFF_UP
        expected.extend_from_slice(&1u32.to_ne_bytes());
        assert_eq!(link_up_body(7), expected);
    }

    #[test]
    fn address_body_is_ip_addr_add_with_no_peer() {
        // ifaddrmsg (AF_INET=2, /24, flags 0, universe), index 3,
        // IFA_LOCAL=2 then IFA_ADDRESS=1, each 8 bytes (4 header + 4 address)
        let mut expected = vec![2u8, 24, 0, 0];
        expected.extend_from_slice(&3i32.to_ne_bytes());
        for kind in [2u16, 1] {
            expected.extend_from_slice(&ne16(8));
            expected.extend_from_slice(&ne16(kind));
            expected.extend_from_slice(&[192, 168, 127, 2]);
        }
        assert_eq!(address_body(3, [192, 168, 127, 2], 24), expected);
    }

    #[test]
    fn default_route_body_is_ip_route_add_default_via() {
        // rtmsg: AF_INET, dst/src len 0, tos 0, main table 254, RTPROT_BOOT 3,
        // universe 0, RTN_UNICAST 1, flags 0; then RTA_GATEWAY=5
        let mut expected = vec![2u8, 0, 0, 0, 254, 3, 0, 1, 0, 0, 0, 0];
        expected.extend_from_slice(&ne16(8));
        expected.extend_from_slice(&ne16(5));
        expected.extend_from_slice(&[192, 168, 127, 1]);
        assert_eq!(default_route_body([192, 168, 127, 1]), expected);
    }

    #[test]
    fn route_attrs_are_padded_to_four_bytes() {
        let mut body = Vec::new();
        push_route_attr(&mut body, 6, &[1, 2, 3, 4, 5]);
        // rta_len counts the header and the data, not the padding
        let mut expected = [ne16(9), ne16(6)].concat();
        expected.extend_from_slice(&[1, 2, 3, 4, 5, 0, 0, 0]);
        assert_eq!(body, expected);
    }

    /// An NLMSG_ERROR message for `seq` with `error`, and a request header after.
    fn ack(seq: u32, error: i32) -> Vec<u8> {
        let mut message = Vec::new();
        message.extend_from_slice(&36u32.to_ne_bytes());
        message.extend_from_slice(&(libc::NLMSG_ERROR as u16).to_ne_bytes());
        message.extend_from_slice(&0u16.to_ne_bytes());
        message.extend_from_slice(&seq.to_ne_bytes());
        message.extend_from_slice(&0u32.to_ne_bytes());
        message.extend_from_slice(&error.to_ne_bytes());
        message.extend_from_slice(&[0u8; 16]);
        message
    }

    #[test]
    fn netlink_ack_reads_success_and_kernel_errors_of_its_request_only() {
        assert!(matches!(netlink_ack(&ack(4, 0), 4), Some(Ok(()))));
        let error = netlink_ack(&ack(4, -libc::EEXIST), 4).unwrap().unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EEXIST));
        // The ack of another request is skipped; the next message is read
        assert!(netlink_ack(&ack(3, 0), 4).is_none());
        let both = [ack(3, -libc::EINVAL), ack(4, 0)].concat();
        assert!(matches!(netlink_ack(&both, 4), Some(Ok(()))));
        // A cut message is not read past its end
        assert!(netlink_ack(&ack(4, 0)[..18], 4).is_none());
    }

    #[test]
    fn link_up_of_loopback_gets_an_ack_from_the_kernel() {
        // The full path: socket, request, ack. Without CAP_NET_ADMIN (a test
        // container), the kernel refuses with EPERM; lo is up, so with it: Ok.
        let rtnl = Rtnetlink::open().unwrap();
        match rtnl.link_up(c"lo") {
            Ok(()) => {}
            Err(e) => assert_eq!(e.raw_os_error(), Some(libc::EPERM), "{e}"),
        }
        let missing = rtnl.link_up(c"nosuchif0").unwrap_err();
        assert_eq!(missing.raw_os_error(), Some(libc::ENODEV));
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
        // ZRAM_READY and NETWORK_STATE stay unset in tests: if the zram check
        // did not run first, this call would wait for the network instead.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            wait_for_untrusted_readiness(),
        )
        .await
        .expect("the zram check must not wait for the network");
        assert_eq!(result, Err("zram safety setup failed"));
    }
}

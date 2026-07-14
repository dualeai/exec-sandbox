//! Memory-pressure guard for the capped zram swap device.
//!
//! A zram device can still advertise logical swap slots after its physical
//! `mem_limit` is full. Under incompressible workloads, direct reclaim may then
//! retry failed swap writes without reaching the kernel OOM killer. The guard
//! watches PSI stall events, confirms the condition with zram counters,
//! and retires every untrusted task in the disposable guest.
//!
//! The monitor's decision path performs no heap allocation: files are opened
//! during setup and pressure counters are parsed from fixed buffers.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicBool, Ordering};

const PSI_PATH: &str = "/proc/pressure/memory";
const MM_STAT_PATH: &str = "/sys/block/zram0/mm_stat";
const DISKSIZE_PATH: &str = "/sys/block/zram0/disksize";

// Fire only when at least one non-idle task is memory-stalled for 80% of a
// 500 ms window. A lower 20% threshold kills bounded workloads that are still
// making legitimate swap-backed progress. `full` is insufficient here: PID1
// or kernel work may remain runnable while the untrusted allocator is trapped
// in direct reclaim. Zram counters are the mandatory second signal.
const PSI_TRIGGER: &[u8] = b"some 400000 500000\n";

static RETIRE_REQUIRED: AtomicBool = AtomicBool::new(false);

/// Refuse new untrusted work once terminal VM retirement has been claimed.
pub(crate) fn ensure_vm_usable() -> io::Result<()> {
    if retirement_required() {
        Err(io::Error::other("VM retirement required"))
    } else {
        Ok(())
    }
}

pub(crate) fn mark_vm_for_retirement() {
    RETIRE_REQUIRED.store(true, Ordering::Release);
}

pub(crate) fn retirement_required() -> bool {
    RETIRE_REQUIRED.load(Ordering::Acquire)
}

/// Reconcile an observed exit code with the terminal-retirement contract.
///
/// Numeric 137 is reserved as the terminal VM signal at the host API boundary:
/// the host retires the session on any 137, whether it came from this guard,
/// the kernel OOM killer, or user code calling `exit(137)`. Conversely, a
/// pending retirement decision overrides any observed code so a non-terminal
/// result is never reported after it. Returns the exit code to report and
/// marks the VM for retirement whenever that code is 137. Callers that only
/// need the marking side effect (e.g. warm-up failure paths that report an
/// error string, not an exit code) may ignore the return value.
pub(crate) fn reconcile_retirement_exit(observed: i32) -> i32 {
    if retirement_required() || observed == 137 {
        mark_vm_for_retirement();
        137
    } else {
        observed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ZramMemoryStats {
    mem_used_total: u64,
    mem_limit: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VerifiedZramGeometry {
    pub(crate) mem_limit: u64,
    pub(crate) disksize: u64,
}

/// Opened and verified resources for the allocation-free decision path.
pub(crate) struct PreparedOomGuard {
    psi: File,
    mm_stat: File,
}

impl PreparedOomGuard {
    /// Verify PID1, register a PSI trigger, and confirm the effective zram cap.
    pub(crate) fn prepare(
        expected_mem_limit: u64,
        expected_disksize: u64,
    ) -> io::Result<(Self, VerifiedZramGeometry)> {
        if std::process::id() != 1 {
            return Err(io::Error::other(
                "OOM guard requires guest-agent to run as PID 1",
            ));
        }

        let mut psi = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(PSI_PATH)?;
        psi.write_all(PSI_TRIGGER)?;

        let mm_stat = File::open(MM_STAT_PATH)?;
        let disksize_file = File::open(DISKSIZE_PATH)?;
        let memory = read_memory_stats(&mm_stat)?;
        let disksize = read_single_u64(&disksize_file)?;

        let geometry =
            verify_zram_geometry_values(memory, disksize, expected_mem_limit, expected_disksize)?;
        Ok((Self { psi, mm_stat }, geometry))
    }

    /// Start the monitor after swapon has succeeded.
    pub(crate) fn spawn(self) -> io::Result<()> {
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(0);
        std::thread::Builder::new()
            .name("oom-pressure".into())
            .stack_size(64 * 1024)
            .spawn(move || match pin_monitor_thread() {
                Ok(()) => {
                    let _ = ready_tx.send(Ok(()));
                    self.run()
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                }
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err((operation, errno))) => Err(io::Error::other(format!(
                "{operation}: {}",
                io::Error::from_raw_os_error(errno)
            ))),
            Err(_) => Err(io::Error::other("OOM guard thread exited during setup")),
        }
    }

    fn run(self) {
        let psi_fd = self.psi.as_raw_fd();
        let mut poll_fd = libc::pollfd {
            fd: psi_fd,
            events: libc::POLLPRI,
            revents: 0,
        };
        let mut mm_buf = [0u8; 256];
        loop {
            poll_fd.revents = 0;
            let rc = unsafe { libc::poll(std::ptr::from_mut(&mut poll_fd), 1, -1) };
            if rc < 0 {
                if io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                monitor_failed();
            }
            // Infinite poll timeout (-1): rc is never 0, only <0 or ready fds.
            // With events == POLLPRI, poll(2) masks revents to
            // {POLLPRI, POLLERR, POLLHUP, POLLNVAL}, so "PRI set and no error
            // bit" is exactly revents == POLLPRI.
            if poll_fd.revents != libc::POLLPRI {
                monitor_failed();
            }

            let memory = read_memory_stats_into(&self.mm_stat, &mut mm_buf)
                .unwrap_or_else(|_| monitor_failed());
            if !capped_zram_pressure(memory) {
                continue;
            }
            mark_vm_for_retirement();
            retire_untrusted_tasks();
            return;
        }
    }
}

/// Terminate every guest task except PID 1 so the normal execution protocol
/// can report exit 137 before the host confirms VM teardown.
///
/// `prepare` requires this process to be Linux PID 1. Linux excludes PID 1 and
/// the calling process from `kill(-1, sig)`, while still reaching process-group
/// leaders and detached descendants. If signaling fails for any reason other
/// than the workload already being gone, the guest can no longer uphold the
/// pressure-containment invariant and must power off.
fn retire_untrusted_tasks() {
    if unsafe { libc::kill(-1, libc::SIGKILL) } == 0 {
        return;
    }
    if last_errno() != libc::ESRCH {
        emergency_poweroff();
    }
}

fn capped_zram_pressure(memory: ZramMemoryStats) -> bool {
    // Zram accounting includes allocator overhead, so physical usage need not
    // land exactly on mem_limit before progress stops. The PSI trigger supplies
    // the stall signal; this predicate confirms substantial physical occupancy.
    // Threshold: mem_limit - mem_limit/4 = ceil(0.75 * mem_limit).
    memory.mem_limit > 0 && memory.mem_used_total >= memory.mem_limit - memory.mem_limit / 4
}

fn verify_zram_geometry_values(
    memory: ZramMemoryStats,
    disksize: u64,
    expected_mem_limit: u64,
    expected_disksize: u64,
) -> io::Result<VerifiedZramGeometry> {
    let page_size = page_size()?;
    // The kernel PAGE_ALIGNs both values up; the effective value must equal the
    // rounded request. Keep the `== 0` term: it is not subsumed by the equality
    // when a requested value is 0 (both round to 0 and would otherwise pass).
    fn check_rounded(
        name: &str,
        effective: u64,
        requested: u64,
        page_size: u64,
    ) -> io::Result<u64> {
        // Values are ≤40% of guest RAM; u64 rounding arithmetic cannot overflow.
        let expected = requested.div_ceil(page_size) * page_size;
        if effective == 0 || effective != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "zram {name} mismatch: requested {requested}, expected kernel-rounded {expected}, effective {effective}"
                ),
            ));
        }
        Ok(effective)
    }
    Ok(VerifiedZramGeometry {
        mem_limit: check_rounded("mem_limit", memory.mem_limit, expected_mem_limit, page_size)?,
        disksize: check_rounded("disksize", disksize, expected_disksize, page_size)?,
    })
}

/// Keep the monitor runnable when the only vCPU is dominated by direct reclaim.
/// Its bounded loop always blocks in poll, so elevated priority cannot create a
/// busy loop. Locking current PID1 mappings prevents the agent or monitor stack
/// from faulting through zram while the device itself rejects writes.
fn pin_monitor_thread() -> Result<(), (&'static str, i32)> {
    if unsafe { libc::mlockall(libc::MCL_CURRENT) } != 0 {
        return Err(("mlockall", last_errno()));
    }

    // Highest non-real-time priority: enough to stay runnable under direct
    // reclaim without risking RT starvation of the single vCPU. Linux nice
    // values are per-thread, so this elevates only the bounded monitor thread.
    if unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, -20) } != 0 {
        return Err(("setpriority", last_errno()));
    }
    Ok(())
}

fn last_errno() -> i32 {
    io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

pub(crate) fn page_size() -> io::Result<u64> {
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if value <= 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(value as u64)
    }
}

fn read_memory_stats(file: &File) -> io::Result<ZramMemoryStats> {
    let mut buf = [0u8; 256];
    read_memory_stats_into(file, &mut buf)
}

fn read_single_u64(file: &File) -> io::Result<u64> {
    let mut buf = [0u8; 64];
    let n = read_fd_at_start(file, &mut buf)?;
    parse_nth_u64(&buf[..n], 0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid integer sysfs value"))
}

fn read_memory_stats_into(file: &File, buf: &mut [u8]) -> io::Result<ZramMemoryStats> {
    let n = read_fd_at_start(file, buf)?;
    parse_memory_stats(&buf[..n])
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid zram mm_stat"))
}

// `Seek`/`Read` are implemented for `&File`, so the shared handle needs no
// `&mut` and the path stays allocation-free. EINTR must be retried, not
// treated as failure: in the monitor loop any error powers the VM off, at
// setup time it marks the VM unavailable, and a signal delivered to PID 1
// mid-read is routine (the poll path retries it too).
fn read_fd_at_start(mut file: &File, buf: &mut [u8]) -> io::Result<usize> {
    loop {
        match file.rewind() {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
        match file.read(buf) {
            Ok(n) => return Ok(n),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
}

fn parse_memory_stats(input: &[u8]) -> Option<ZramMemoryStats> {
    // mm_stat: orig_data_size compr_data_size mem_used_total mem_limit ...
    let mut fields = std::str::from_utf8(input).ok()?.split_ascii_whitespace();
    Some(ZramMemoryStats {
        mem_used_total: fields.nth(2)?.parse().ok()?,
        mem_limit: fields.next()?.parse().ok()?,
    })
}

fn parse_nth_u64(input: &[u8], wanted: usize) -> Option<u64> {
    // split_ascii_whitespace yields borrowed sub-slices (no allocation); u64
    // FromStr does not allocate — the decision path stays allocation-free.
    std::str::from_utf8(input)
        .ok()?
        .split_ascii_whitespace()
        .nth(wanted)?
        .parse()
        .ok()
}

fn monitor_failed() -> ! {
    emergency_poweroff();
}

/// A runtime monitor failure invalidates the capped-zram safety invariant.
/// Power off immediately; if reboot(2) is unavailable, exiting PID1 leaves the
/// host liveness timeout to destroy the unusable VM.
pub(crate) fn emergency_poweroff() -> ! {
    unsafe {
        libc::reboot(libc::LINUX_REBOOT_CMD_POWER_OFF);
        libc::_exit(70);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_current_mm_stat_layout() {
        let stats = parse_memory_stats(b"8192 4096 12288 41943040 16384 1 0 0 0\n").unwrap();
        assert_eq!(stats.mem_used_total, 12288);
        assert_eq!(stats.mem_limit, 41_943_040);
    }

    #[test]
    fn rejects_missing_or_non_numeric_fields() {
        assert_eq!(parse_memory_stats(b"1 2 3\n"), None);
        assert_eq!(parse_memory_stats(b"1 2 invalid 4\n"), None);
    }

    #[test]
    fn reconcile_retirement_exit_truth_table() {
        // RETIRE_REQUIRED is a process-global sticky flag, so the three cases
        // run in one test, in an order that only ever flips it false -> true.
        assert!(!retirement_required());
        assert_eq!(reconcile_retirement_exit(0), 0);
        assert_eq!(reconcile_retirement_exit(1), 1);
        assert!(!retirement_required()); // non-137 codes never mark

        assert_eq!(reconcile_retirement_exit(137), 137);
        assert!(retirement_required()); // observed 137 marks retirement

        // Pending retirement overrides any observed code.
        assert_eq!(reconcile_retirement_exit(0), 137);
        assert_eq!(reconcile_retirement_exit(42), 137);
    }

    #[test]
    fn pressure_threshold_is_inclusive_for_physical_zram_usage() {
        let below = ZramMemoryStats {
            mem_used_total: 74,
            mem_limit: 100,
        };
        assert!(!capped_zram_pressure(below));

        let physical = ZramMemoryStats {
            mem_used_total: 75,
            ..below
        };
        assert!(capped_zram_pressure(physical));

        // Non-multiple-of-4 limit: threshold is ceil(0.75 * 101) = 76.
        let below_odd = ZramMemoryStats {
            mem_used_total: 75,
            mem_limit: 101,
        };
        assert!(!capped_zram_pressure(below_odd));
        let at_odd = ZramMemoryStats {
            mem_used_total: 76,
            ..below_odd
        };
        assert!(capped_zram_pressure(at_odd));
    }

    #[test]
    fn geometry_verification_accepts_page_rounded_values() {
        let page = page_size().unwrap();
        let requested_mem_limit = page + 1;
        let requested_disksize = page * 2 + 1;
        let geometry = verify_zram_geometry_values(
            ZramMemoryStats {
                mem_used_total: 0,
                mem_limit: page * 2,
            },
            page * 3,
            requested_mem_limit,
            requested_disksize,
        )
        .unwrap();

        assert_eq!(geometry.mem_limit, page * 2);
        assert_eq!(geometry.disksize, page * 3);
    }

    #[test]
    fn geometry_verification_rejects_zero_and_mismatch() {
        let page = page_size().unwrap();
        let stats = |mem_limit| ZramMemoryStats {
            mem_used_total: 0,
            mem_limit,
        };

        assert!(verify_zram_geometry_values(stats(0), page, page, page).is_err());
        assert!(verify_zram_geometry_values(stats(page * 2), page, page, page).is_err());
        assert!(verify_zram_geometry_values(stats(page), 0, page, page).is_err());
        assert!(verify_zram_geometry_values(stats(page), page * 2, page, page).is_err());
    }
}

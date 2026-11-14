//! Per-request wall clock sync from the host.
//!
//! Every request may carry `host_time_ns` (host CLOCK_REALTIME at send time).
//! After an L1 restore, a load or a long pause the guest clock lags by the time
//! the VM was stopped; the next request steps it forward. It never steps back:
//! a request can wait in the queue behind a long exec, so its stamp may be old.

/// Step forward when the guest lags the host by more than this.
const STEP_FORWARD_NS: i128 = 100_000_000; // 100 ms

/// New guest time, or None when the guest clock is close enough or ahead.
pub(crate) fn step_target(guest_ns: i128, host_ns: i128) -> Option<i128> {
    (host_ns - guest_ns > STEP_FORWARD_NS).then_some(host_ns)
}

/// Guest CLOCK_REALTIME in nanoseconds since the epoch.
fn realtime_ns() -> i128 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts) };
    i128::from(ts.tv_sec) * 1_000_000_000 + i128::from(ts.tv_nsec)
}

/// Step the guest wall clock forward to the host time carried by a request.
pub(crate) fn sync(host_time_ns: Option<u64>) {
    let Some(host_ns) = host_time_ns else {
        return;
    };
    let guest_ns = realtime_ns();
    let Some(target) = step_target(guest_ns, i128::from(host_ns)) else {
        return;
    };
    let ts = libc::timespec {
        tv_sec: (target / 1_000_000_000) as libc::time_t,
        tv_nsec: (target % 1_000_000_000) as libc::c_long,
    };
    if unsafe { libc::clock_settime(libc::CLOCK_REALTIME, &ts) } == 0 {
        log_info!("[clock] stepped by {}ms", (target - guest_ns) / 1_000_000);
    } else {
        log_warn!(
            "[clock] clock_settime failed: {}",
            std::io::Error::last_os_error()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    const S: i128 = 1_000_000_000;

    #[rstest]
    #[case::in_sync(100 * S, 100 * S, None)]
    #[case::small_lag(100 * S, 100 * S + 50_000_000, None)]
    #[case::lag_after_restore(100 * S, 160 * S, Some(160 * S))]
    #[case::just_above_threshold(100 * S, 100 * S + 100_000_001, Some(100 * S + 100_000_001))]
    #[case::stale_stamp_never_steps_back(100 * S, 90 * S, None)]
    #[case::small_lead(100 * S, 100 * S - 500_000_000, None)]
    fn step_rules(#[case] guest: i128, #[case] host: i128, #[case] expected: Option<i128>) {
        assert_eq!(step_target(guest, host), expected);
    }
}

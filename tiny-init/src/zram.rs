//! zram swap setup.

use std::ffi::CString;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::device;
use crate::sys::{self, SWAP_FLAG_PREFER, syscall_nr};

/// Build a swap header for the given device size.
/// Returns a 4096-byte header or None if the size is too small.
fn build_swap_header(device_size: u64) -> Option<Vec<u8>> {
    let pages = (device_size / 4096).saturating_sub(1) as u32;
    if pages < 10 {
        return None; // kernel rejects tiny swap
    }
    let mut header = vec![0u8; 4096];
    // SWAPSPACE2 signature at offset 4086
    header[4086..4096].copy_from_slice(b"SWAPSPACE2");
    // version = 1
    header[1024..1028].copy_from_slice(&1u32.to_le_bytes());
    // last_page (0-indexed: total_pages - 1, since page 0 is the header)
    header[1028..1032].copy_from_slice(&pages.to_le_bytes());
    Some(header)
}

pub(crate) fn setup_zram(kver: &str) {
    log_fmt!("[zram] setup starting (kernel {})", kver);

    let m = format!("/lib/modules/{}/kernel", kver);

    // Load modules with logging
    let lz4_compress_ok = sys::load_module(&format!("{}/lib/lz4/lz4_compress.ko", m), true);
    sys::load_module(&format!("{}/crypto/lz4.ko", m), true);
    let zram_ok = sys::load_module(&format!("{}/drivers/block/zram/zram.ko", m), true);

    if !zram_ok && !lz4_compress_ok {
        log_fmt!("[zram] module load failed, aborting");
        return;
    }

    // Wait for zram device to appear after module load
    // Exponential backoff: 1+2+4+8+16+32+64+128 = 255ms max
    // CI runners with nested virtualization may need longer waits
    log_fmt!("[zram] waiting for /sys/block/zram0...");
    if !device::poll_backoff(
        &[1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000],
        || Path::new("/sys/block/zram0").exists(),
    ) {
        log_fmt!("[zram] device not found, aborting");
        return;
    }

    // Use proper fallback chain: lz4 -> lzo-rle -> lzo
    let algo = ["lz4", "lzo-rle", "lzo"]
        .iter()
        .find(|a| fs::write("/sys/block/zram0/comp_algorithm", a).is_ok());
    match algo {
        Some(a) => log_fmt!("[zram] compression: {}", a),
        None => {
            log_fmt!("[zram] failed to set compression algorithm, aborting");
            return;
        }
    }

    // Kernel 6.16+: algorithm-specific tuning via sysfs.
    // LZ4 "level" is an acceleration parameter (lower = better compression,
    // higher = faster). Level 1 = max compression ratio, best for 128-256MB
    // VMs where every byte of zram matters. CPU cost is negligible vs. the
    // memory savings on small VMs.
    // Must be set AFTER comp_algorithm but BEFORE disksize.
    // Ref: https://docs.kernel.org/admin-guide/blockdev/zram.html
    let _ = fs::write("/sys/block/zram0/algorithm_params", "level=1");

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
    let header = match build_swap_header(zram_size) {
        Some(h) => h,
        None => {
            log_fmt!("[zram] device too small for swap, aborting");
            return;
        }
    };
    let header_result = (|| -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new().write(true).open("/dev/zram0")?;
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
        log_fmt!(
            "[zram] swapon failed (errno={}), aborting",
            sys::last_errno()
        );
        return;
    }

    log_fmt!("[zram] swap enabled");

    // Security: remove device node after swapon (kernel holds internal reference).
    device::remove_device_node("/dev/zram0");

    // VM tuning (these can fail silently - non-critical)
    let _ = fs::write("/proc/sys/vm/page-cluster", "0");
    let _ = fs::write("/proc/sys/vm/swappiness", "180");
    let _ = fs::write(
        "/proc/sys/vm/min_free_kbytes",
        (mem_kb * 4 / 100).to_string(),
    );
    let _ = fs::write("/proc/sys/vm/overcommit_memory", "0");

    // Kernel 6.15+: proactive defragmentation mode.
    // Mode 1 = the page allocator groups allocations by mobility type more
    // aggressively, preserving contiguous free regions for large folios.
    // Critical for 128-256MB VMs where fragmentation quickly exhausts
    // contiguous memory and degrades tmpfs/ext4 large folio performance.
    // Ref: https://kernelnewbies.org/Linux_6.15 (vm.defrag_mode)
    let _ = fs::write("/proc/sys/vm/defrag_mode", "1");

    log_fmt!("[zram] setup complete");
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn swap_header_signature() {
        let h = build_swap_header(1024 * 1024).unwrap(); // 1MB
        assert_eq!(&h[4086..4096], b"SWAPSPACE2");
    }

    #[test]
    fn swap_header_version() {
        let h = build_swap_header(1024 * 1024).unwrap();
        assert_eq!(u32::from_le_bytes(h[1024..1028].try_into().unwrap()), 1);
    }

    #[test]
    fn swap_header_pages() {
        let h = build_swap_header(1024 * 1024).unwrap(); // 1MB = 256 pages
        let pages = u32::from_le_bytes(h[1028..1032].try_into().unwrap());
        assert_eq!(pages, 255); // 256 - 1 (page 0 is header)
    }

    #[test]
    fn swap_header_rejects_tiny() {
        assert!(build_swap_header(4096 * 5).is_none()); // 5 pages < 10
        assert!(build_swap_header(0).is_none());
    }

    #[test]
    fn swap_header_minimum_valid() {
        assert!(build_swap_header(4096 * 11).is_some()); // 11 pages, last_page=10 >= 10
    }

    proptest! {
        #[test]
        fn swap_header_pages_never_overflow(mem_kb in 1u64..=16_777_216) {
            let zram_size = mem_kb * 512;
            if let Some(h) = build_swap_header(zram_size) {
                let pages = u32::from_le_bytes(h[1028..1032].try_into().unwrap());
                // Total pages represented must not exceed device size
                prop_assert!((pages as u64 + 1) * 4096 <= zram_size + 4096);
                // Must have at least 10 pages
                prop_assert!(pages >= 10);
            }
        }
    }
}

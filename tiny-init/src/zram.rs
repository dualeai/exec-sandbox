//! zram swap header utilities.

/// Build a swap header for the given device size.
/// Returns a PAGE_SIZE-byte header or None if the size is too small.
///
/// Uses runtime sysconf(_SC_PAGESIZE) to support both 4KB (x86_64)
/// and 16KB (aarch64 with CONFIG_ARM64_16K_PAGES) kernels.
/// NOTE: Mirrored in guest-agent/src/init.rs — keep both in sync.
#[cfg(test)]
fn build_swap_header(device_size: u64) -> Option<Vec<u8>> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
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
    // last_page (0-indexed: total_pages - 1, since page 0 is the header)
    header[1028..1032].copy_from_slice(&pages.to_le_bytes());
    Some(header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn host_page_size() -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    #[test]
    fn swap_header_signature() {
        let ps = host_page_size();
        let h = build_swap_header(1024 * 1024).unwrap(); // 1MB
        assert_eq!(&h[ps - 10..ps], b"SWAPSPACE2");
    }

    #[test]
    fn swap_header_version() {
        let h = build_swap_header(1024 * 1024).unwrap();
        assert_eq!(u32::from_le_bytes(h[1024..1028].try_into().unwrap()), 1);
    }

    #[test]
    fn swap_header_pages() {
        let ps = host_page_size() as u64;
        let h = build_swap_header(1024 * 1024).unwrap();
        let pages = u32::from_le_bytes(h[1028..1032].try_into().unwrap());
        let expected = (1024 * 1024 / ps - 1) as u32;
        assert_eq!(pages, expected);
    }

    #[test]
    fn swap_header_rejects_tiny() {
        let ps = host_page_size() as u64;
        assert!(build_swap_header(ps * 5).is_none()); // 5 pages < 10
        assert!(build_swap_header(0).is_none());
    }

    #[test]
    fn swap_header_minimum_valid() {
        let ps = host_page_size() as u64;
        assert!(build_swap_header(ps * 11).is_some()); // 11 pages, last_page=10 >= 10
    }

    proptest! {
        #[test]
        fn swap_header_pages_never_overflow(mem_kb in 1u64..=16_777_216) {
            let ps = host_page_size() as u64;
            let zram_size = mem_kb * 512;
            if let Some(h) = build_swap_header(zram_size) {
                let pages = u32::from_le_bytes(h[1028..1032].try_into().unwrap());
                // Total pages represented must not exceed device size
                prop_assert!((pages as u64 + 1) * ps <= zram_size + ps);
                // Must have at least 10 pages
                prop_assert!(pages >= 10);
            }
        }
    }
}

//! zram swap header utilities.

/// Build a swap header for the given device size.
/// Returns a 4096-byte header or None if the size is too small.
#[allow(dead_code)] // Used by tests only; device setup moved to guest-agent
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

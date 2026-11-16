//! The kernel random source: reseed the CRNG, after a mix of the host's bytes when a
//! Ping carries them.
//!
//! RNDRESEEDCRNG makes the CRNG take a new key from the input pool at once. Without it,
//! the kernel reseeds only at its interval: max(1 s, uptime / 2) in the first 120 s of
//! uptime, then 60 s (CRNG_RESEED_INTERVAL); the virtio-rng thread adds its bytes only
//! after the same sleep. On HVF the input pool gets no fresh secret bytes at a reseed
//! (no RNDR, no SMCCC TRNG: only counter reads); the host's bytes of a Ping are such
//! bytes. `mix_and_reseed` adds them to the pool (credited), then reseeds; the Ping gets
//! an error, not a Pong, when a step fails (connection.rs).
//!
//! `reseed` runs at each port open (at boot, and after a port error) and before each
//! command but a file chunk; a failure there only logs.
//!
//! Cost: ~11 µs per `reseed` (open, RNDRESEEDCRNG, close).

use std::io;
use std::os::fd::AsRawFd;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

/// Bytes of host entropy in a Ping (`host_entropy`, base64).
const HOST_ENTROPY_BYTES: usize = 32;

/// The two kernel operations of `mix_and_reseed`. A trait, so that the tests check
/// their order and failures with a fake: both ioctls need CAP_SYS_ADMIN.
trait KernelRandom {
    /// Add `bytes` to the input pool and credit their 256 bits (RNDADDENTROPY). With the
    /// credit, a CRNG that is not ready yet gets ready, and the reseed then works.
    fn mix(&self, bytes: &[u8; HOST_ENTROPY_BYTES]) -> io::Result<()>;
    /// Make the CRNG take a new key from the input pool (RNDRESEEDCRNG).
    fn reseed(&self) -> io::Result<()>;
}

// ioctl request numbers of include/uapi/linux/random.h, in the generic _IOC encoding of
// x86_64 and aarch64. libc::ioctl takes a c_ulong on glibc and a c_int on musl: both
// values are positive c_int values, so `as _` keeps them the same on both.
/// RNDADDENTROPY = _IOW('R', 0x03, int[2])
const RNDADDENTROPY: libc::c_int = 0x4008_5203;
/// RNDRESEEDCRNG = _IO('R', 0x07)
const RNDRESEEDCRNG: libc::c_int = 0x5207;

/// struct rand_pool_info of RNDADDENTROPY, with a buffer of HOST_ENTROPY_BYTES.
#[repr(C)]
struct RandPoolInfo {
    /// Bits to credit.
    entropy_count: libc::c_int,
    /// Bytes in `buf`.
    buf_size: libc::c_int,
    /// The host bytes (the kernel struct types them `__u32[]`, and reads them as bytes).
    buf: [u8; HOST_ENTROPY_BYTES],
}

/// The RNDADDENTROPY argument for `bytes`: all of them, with 8 bits credited for each.
fn rand_pool_info(bytes: &[u8; HOST_ENTROPY_BYTES]) -> RandPoolInfo {
    RandPoolInfo {
        entropy_count: (HOST_ENTROPY_BYTES * 8) as libc::c_int,
        buf_size: HOST_ENTROPY_BYTES as libc::c_int,
        buf: *bytes,
    }
}

/// /dev/urandom, open for the two ioctls.
struct DevUrandom(std::fs::File);

impl DevUrandom {
    /// Open /dev/urandom, write only.
    fn open() -> io::Result<Self> {
        std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/urandom")
            .map(Self)
    }
}

/// Err(the errno) when an ioctl returned a negative value.
fn ioctl_result(ret: libc::c_int) -> io::Result<()> {
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

impl KernelRandom for DevUrandom {
    fn mix(&self, bytes: &[u8; HOST_ENTROPY_BYTES]) -> io::Result<()> {
        let info = rand_pool_info(bytes);
        // SAFETY: the descriptor is open, and `info` holds the two ints and the
        // `buf_size` bytes that the kernel reads.
        ioctl_result(unsafe {
            libc::ioctl(self.0.as_raw_fd(), RNDADDENTROPY as _, &raw const info)
        })
    }

    fn reseed(&self) -> io::Result<()> {
        // SAFETY: the descriptor is open; the request takes no argument.
        ioctl_result(unsafe { libc::ioctl(self.0.as_raw_fd(), RNDRESEEDCRNG as _) })
    }
}

/// Add the host entropy of a Ping to the input pool, then reseed the CRNG. Err names the
/// step that failed; no step runs after it.
fn mix_and_reseed(kernel: &impl KernelRandom, host_entropy: &str) -> Result<(), String> {
    let decoded = BASE64
        .decode(host_entropy)
        .map_err(|e| format!("host entropy is not base64: {e}"))?;
    let bytes: [u8; HOST_ENTROPY_BYTES] = decoded.as_slice().try_into().map_err(|_| {
        format!(
            "host entropy has {} bytes, not {HOST_ENTROPY_BYTES}",
            decoded.len()
        )
    })?;
    kernel
        .mix(&bytes)
        .map_err(|e| format!("ioctl RNDADDENTROPY failed: {e}"))?;
    kernel
        .reseed()
        .map_err(|e| format!("ioctl RNDRESEEDCRNG failed: {e}"))
}

/// `mix_and_reseed` on /dev/urandom.
pub(crate) fn mix_host_entropy(host_entropy: &str) -> Result<(), String> {
    let kernel = DevUrandom::open().map_err(|e| format!("open /dev/urandom failed: {e}"))?;
    mix_and_reseed(&kernel, host_entropy)
}

/// Reseed the CRNG from its input pool. A failure only logs (the reseed of a Ping with
/// host entropy answers an error instead: connection.rs).
pub(crate) fn reseed() {
    match DevUrandom::open() {
        Ok(kernel) => match kernel.reseed() {
            Ok(()) => log_info!("Kernel CRNG reseeded"),
            Err(e) => log_warn!("Kernel CRNG reseed: ioctl RNDRESEEDCRNG failed: {e}"),
        },
        Err(e) => log_warn!("Kernel CRNG reseed: open /dev/urandom failed: {e}"),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A KernelRandom that records its calls and fails the steps it is told to.
    #[derive(Default)]
    struct FakeKernel {
        calls: RefCell<Vec<String>>,
        mix_fails: bool,
        reseed_fails: bool,
    }

    impl KernelRandom for FakeKernel {
        fn mix(&self, bytes: &[u8; HOST_ENTROPY_BYTES]) -> io::Result<()> {
            self.calls.borrow_mut().push(format!("mix {bytes:?}"));
            if self.mix_fails {
                return Err(io::Error::from_raw_os_error(libc::EPERM));
            }
            Ok(())
        }

        fn reseed(&self) -> io::Result<()> {
            self.calls.borrow_mut().push("reseed".to_string());
            if self.reseed_fails {
                return Err(io::Error::from_raw_os_error(libc::ENODATA));
            }
            Ok(())
        }
    }

    /// Base64 of 32 bytes (0xfb, 0xef, 0xff, then 3, 4, ..., 31: the text starts with "++//",
    /// two characters of the standard alphabet only), and the bytes.
    fn entropy_of_32_bytes() -> (String, [u8; HOST_ENTROPY_BYTES]) {
        let bytes: [u8; HOST_ENTROPY_BYTES] = std::array::from_fn(|i| match i {
            0 => 0xfb,
            1 => 0xef,
            2 => 0xff,
            _ => i as u8,
        });
        let encoded = BASE64.encode(bytes);
        assert!(encoded.starts_with("++//"), "{encoded}");
        (encoded, bytes)
    }

    #[test]
    fn the_rndaddentropy_argument_holds_the_32_bytes_and_credits_256_bits() {
        let (_, bytes) = entropy_of_32_bytes();

        let info = rand_pool_info(&bytes);

        assert_eq!(
            (info.entropy_count, info.buf_size, info.buf),
            (256, 32, bytes)
        );
    }

    #[test]
    fn mix_and_reseed_mixes_the_host_bytes_then_reseeds() {
        let kernel = FakeKernel::default();
        let (encoded, bytes) = entropy_of_32_bytes();

        assert_eq!(mix_and_reseed(&kernel, &encoded), Ok(()));
        assert_eq!(
            *kernel.calls.borrow(),
            vec![format!("mix {bytes:?}"), "reseed".to_string()]
        );
    }

    #[test]
    fn host_entropy_that_is_not_base64_reaches_no_kernel_step() {
        let kernel = FakeKernel::default();

        let error = mix_and_reseed(&kernel, "not base64!").unwrap_err();

        assert!(error.starts_with("host entropy is not base64: "), "{error}");
        assert!(kernel.calls.borrow().is_empty());
    }

    #[rstest::rstest]
    #[case(31)]
    #[case(33)]
    #[case(0)]
    fn host_entropy_of_another_length_reaches_no_kernel_step(#[case] length: usize) {
        let kernel = FakeKernel::default();

        let error = mix_and_reseed(&kernel, &BASE64.encode(vec![7u8; length])).unwrap_err();

        assert_eq!(error, format!("host entropy has {length} bytes, not 32"));
        assert!(kernel.calls.borrow().is_empty());
    }

    #[test]
    fn a_failed_mix_names_its_step_and_skips_the_reseed() {
        let kernel = FakeKernel {
            mix_fails: true,
            ..FakeKernel::default()
        };
        let (encoded, bytes) = entropy_of_32_bytes();

        let error = mix_and_reseed(&kernel, &encoded).unwrap_err();

        assert_eq!(
            error,
            format!(
                "ioctl RNDADDENTROPY failed: {}",
                io::Error::from_raw_os_error(libc::EPERM)
            )
        );
        assert_eq!(*kernel.calls.borrow(), vec![format!("mix {bytes:?}")]);
    }

    #[test]
    fn a_failed_reseed_names_its_step() {
        let kernel = FakeKernel {
            reseed_fails: true,
            ..FakeKernel::default()
        };
        let (encoded, bytes) = entropy_of_32_bytes();

        let error = mix_and_reseed(&kernel, &encoded).unwrap_err();

        assert_eq!(
            error,
            format!(
                "ioctl RNDRESEEDCRNG failed: {}",
                io::Error::from_raw_os_error(libc::ENODATA)
            )
        );
        assert_eq!(
            *kernel.calls.borrow(),
            vec![format!("mix {bytes:?}"), "reseed".to_string()]
        );
    }

    #[test]
    fn rand_pool_info_has_the_layout_of_the_kernel_struct() {
        // Two ints, then the buffer at offset 8: the kernel reads buf_size bytes there.
        assert_eq!(std::mem::offset_of!(RandPoolInfo, entropy_count), 0);
        assert_eq!(std::mem::offset_of!(RandPoolInfo, buf_size), 4);
        assert_eq!(std::mem::offset_of!(RandPoolInfo, buf), 8);
        assert_eq!(std::mem::size_of::<RandPoolInfo>(), 8 + HOST_ENTROPY_BYTES);
    }

    #[test]
    fn ioctl_numbers_are_those_of_the_kernel_header() {
        // _IOC(dir, type, nr, size) = dir << 30 | size << 16 | type << 8 | nr
        let ioc =
            |dir: u32, nr: u32, size: u32| (dir << 30) | (size << 16) | (u32::from(b'R') << 8) | nr;
        assert_eq!(RNDADDENTROPY as u32, ioc(1, 0x03, 8)); // _IOW, int[2]
        assert_eq!(RNDRESEEDCRNG as u32, ioc(0, 0x07, 0)); // _IO
    }
}

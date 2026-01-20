# Build Scripts

This directory contains scripts for building VM images and components.

## Quick Reference

| Script | Purpose | Output |
|--------|---------|--------|
| `build-guest-agent.sh` | Build guest-agent binary | `images/dist/guest-agent-linux-{arch}` |
| `build-tiny-init.sh` | Build tiny-init (initramfs init) | `images/dist/tiny-init-{arch}` |
| `build-initramfs.sh` | Package initramfs with kernel modules | `images/dist/initramfs-{arch}` |
| `build-qcow2.sh` | Build root filesystem images | `images/dist/{variant}-base-{arch}.qcow2` |
| `build-images.sh` | Build everything | All of the above |
| `extract-kernel.sh` | Extract kernel from Alpine | `images/dist/vmlinuz-{arch}` |
| `ci-diagnose.sh` | Diagnose CI failures | (stdout) |

## Build Order

Components must be built in dependency order:

```
guest-agent ────► qcow2 images

tiny-init ──────► initramfs
```

**Dependency chain:**
1. `guest-agent` - Rust binary, embedded in qcow2
2. `tiny-init` - Rust binary, embedded in initramfs
3. `initramfs` - Contains tiny-init + kernel modules (independent of qcow2)
4. `qcow2` - Contains guest-agent + runtime (Python/Node/raw)

**Note:** initramfs and qcow2 are independent - you can rebuild one without the other.

## Architecture Support

All scripts support cross-compilation:

```bash
# Build for specific architecture
./scripts/build-guest-agent.sh aarch64
./scripts/build-guest-agent.sh x86_64

# Build for both
./scripts/build-guest-agent.sh all
```

## Caching

Scripts use content-addressable caching via `.hash` sidecar files:
- `guest-agent-linux-aarch64.hash` contains SHA256 of inputs
- Rebuild only happens when inputs change
- **Force rebuild:** Delete the `.hash` file

```bash
# Force rebuild of all qcow2 images
rm -f images/dist/*.qcow2.hash
./scripts/build-qcow2.sh all aarch64
```

---

## Debugging Guide

### Boot Flow

Understanding the boot sequence helps diagnose issues:

```
QEMU starts
    │
    ▼
Kernel loads initramfs
    │
    ▼
tiny-init runs (PID 1 in initramfs)
    ├── Mounts /dev, /proc, /sys, /tmp
    ├── Loads kernel modules (virtio, ext4, etc.)
    ├── Sets up zram swap
    ├── Mounts /dev/vda (qcow2 root) to /mnt
    ├── switch_root to /mnt
    │
    ▼
guest-agent runs (PID 1 in qcow2 root)
    ├── Sets up network (192.168.127.2)
    ├── Opens virtio-serial ports
    └── Waits for commands from host
```

### Common Issues

#### 1. Exit Code 0x6500 (101) - Kernel Panic

**Symptom:**
```
Kernel panic - not syncing: Attempted to kill init! exitcode=0x00006500
```

**Meaning:** Exit code 101 (0x65) is Rust's panic code, but can also occur when `execv()` fails.

**Debug steps:**
1. Check console output for `[init] execv failed: errno=X`
2. Common errno values:
   - `errno=2` (ENOENT): Binary not found
   - `errno=8` (ENOEXEC): Wrong architecture binary
   - `errno=13` (EACCES): Permission denied

**Most likely cause:** Wrong architecture binary in qcow2 image.

**Fix:**
```bash
# Verify binary architectures
file images/dist/guest-agent-linux-*

# Rebuild with correct architecture
./scripts/build-guest-agent.sh aarch64
rm -f images/dist/*.qcow2.hash
./scripts/build-qcow2.sh all aarch64
```

#### 2. Guest Agent Timeout

**Symptom:**
```
VmBootTimeoutError: Guest agent not ready after 30s
```

**Debug steps:**
1. Check console output in the error message
2. Look for tiny-init errors: `[init] ERROR: ...`
3. Look for execv errors: `[init] execv failed: errno=...`
4. Check if kernel modules loaded (ext4, virtio_blk)

**Common causes:**
- Wrong architecture binary (errno=8)
- Missing guest-agent in qcow2 (errno=2)
- Network not configured (gvproxy issue)
- virtio-serial ports not ready

#### 3. Module Load Failures

**Symptom:**
```
[module] ext4.ko: errno=2
```

**Cause:** Kernel module not found in initramfs.

**Fix:** Rebuild initramfs:
```bash
./scripts/build-initramfs.sh aarch64
```

#### 4. Network Issues

**Symptom:** Package installation fails, DNS not working.

**Debug:**
```bash
# Inside VM (if you can get a shell)
ping 192.168.127.1  # Gateway (gvproxy)
cat /etc/resolv.conf  # Should be 192.168.127.1
```

**Common causes:**
- gvproxy not running on host
- eth0 not configured in guest-agent
- resolv.conf not set in qcow2 image

### Adding Debug Output

#### tiny-init

Use `log_fmt!` macro for output (bypasses Rust stdio issues in early boot):

```rust
log_fmt!("[init] Debug message here");
```

Rebuild after changes:
```bash
./scripts/build-tiny-init.sh aarch64
./scripts/build-initramfs.sh aarch64
```

#### guest-agent

Use `eprintln!` for debug output (goes to VM console):

```rust
eprintln!("[guest-agent] Debug message");
```

Rebuild after changes:
```bash
./scripts/build-guest-agent.sh aarch64
rm -f images/dist/*.qcow2.hash
./scripts/build-qcow2.sh all aarch64
```

### Verifying Binaries

Always verify architecture before debugging further:

```bash
# Check all binaries
file images/dist/guest-agent-linux-*
file images/dist/tiny-init-*

# Expected output for aarch64:
# ELF 64-bit LSB executable, ARM aarch64, ...

# Expected output for x86_64:
# ELF 64-bit LSB pie executable, x86-64, ...
```

### Testing Individual Components

```bash
# Test just the health check (fast)
uv run pytest tests/test_vm_manager.py::TestAllImageTypes::test_vm_health_check_all_images -v

# Test specific image type
uv run pytest tests/test_vm_manager.py -v -k python
```

---

## Historical Issues

### 2026-01: Wrong Architecture Binary (ENOEXEC)

**Issue:** Guest-agent crashed with exit code 101 on aarch64.

**Root cause:** `find_guest_agent()` in `build-qcow2.sh` had a fallback path without architecture verification. A cached x86_64 binary was embedded in the aarch64 qcow2.

**Fix:** Added ELF architecture verification in `find_guest_agent()`.

**Lesson:** Always verify binary architecture, never trust file paths alone.

See `INVESTIGATION.md` for full details.

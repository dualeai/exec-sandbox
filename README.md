# exec-sandbox

Secure code execution in isolated QEMU microVMs. Drop-in Python library for running untrusted Python, JavaScript, and shell code with 6-layer security isolation.

[![CI](https://github.com/dualeai/exec-sandbox/actions/workflows/test.yml/badge.svg)](https://github.com/dualeai/exec-sandbox/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/codecov/c/github/dualeai/exec-sandbox)](https://codecov.io/gh/dualeai/exec-sandbox)
[![PyPI](https://img.shields.io/pypi/v/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![Python](https://img.shields.io/pypi/pyversions/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![License](https://img.shields.io/pypi/l/exec-sandbox)](https://opensource.org/licenses/Apache-2.0)

## Highlights

- **Hardware isolation** - Each execution runs in a dedicated QEMU microVM with KVM/HVF, not containers
- **Fast startup** - 400ms cold boot, 1-2ms with warm pool pre-booting
- **Simple API** - Just `Scheduler` and `run()`, async context manager pattern
- **Streaming output** - Real-time stdout/stderr via callbacks
- **Smart caching** - L1 local + L3 S3 snapshot cache for package installation
- **Network control** - Disabled by default, optional DNS-based domain whitelisting
- **Memory optimization** - zram compression + balloon for ~30% more usable memory, ~80% smaller snapshots

## Installation

```bash
uv add exec-sandbox              # Core library
uv add "exec-sandbox[s3]"        # + S3 snapshot caching
```

```bash
# Install QEMU runtime
brew install qemu                # macOS
apt install qemu-system          # Ubuntu/Debian
```

```bash
# Download pre-built VM images
curl -LO https://github.com/dualeai/exec-sandbox/releases/latest/download/images-x86_64.tar.zst
mkdir -p ~/.local/share/exec-sandbox/images
tar --zstd -xf images-x86_64.tar.zst -C ~/.local/share/exec-sandbox/images/
```

## Quick Start

### Basic Execution

```python
from exec_sandbox import Scheduler

async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="print('Hello, World!')",
        language="python",  # or "javascript", "raw"
    )
    print(result.stdout)     # Hello, World!
    print(result.exit_code)  # 0
```

### With Packages

First run installs and creates snapshot; subsequent runs restore in <400ms.

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="import pandas; print(pandas.__version__)",
        language="python",
        packages=["pandas==2.2.0", "numpy==1.26.0"],
    )
    print(result.stdout)  # 2.2.0
```

### Streaming Output

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="for i in range(5): print(i)",
        language="python",
        on_stdout=lambda chunk: print(f"[OUT] {chunk}", end=""),
        on_stderr=lambda chunk: print(f"[ERR] {chunk}", end=""),
    )
```

### Network Access

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="import urllib.request; print(urllib.request.urlopen('https://httpbin.org/ip').read())",
        language="python",
        allow_network=True,
        allowed_domains=["httpbin.org"],  # DNS whitelisting
    )
```

### Production Configuration

```python
from exec_sandbox import Scheduler, SchedulerConfig

config = SchedulerConfig(
    max_concurrent_vms=20,       # Backpressure control
    warm_pool_size=1,            # Enable warm pool (size = max_concurrent_vms × 25%)
    default_memory_mb=512,       # Per-VM memory
    default_timeout_seconds=60,  # Execution timeout
    s3_bucket="my-snapshots",    # L3 cache for package snapshots
    s3_region="us-east-1",
)

async with Scheduler(config) as scheduler:
    result = await scheduler.run(...)
```

### Error Handling

```python
from exec_sandbox import Scheduler, VmTimeoutError, PackageNotAllowedError, SandboxError

async with Scheduler() as scheduler:
    try:
        result = await scheduler.run(code="while True: pass", language="python", timeout_seconds=5)
    except VmTimeoutError:
        print("Execution timed out")
    except PackageNotAllowedError as e:
        print(f"Package not in allowlist: {e}")
    except SandboxError as e:
        print(f"Sandbox error: {e}")
```

## Documentation

- [QEMU Documentation](https://www.qemu.org/docs/master/) - Virtual machine emulator
- [KVM](https://www.linux-kvm.org/page/Documents) - Linux kernel virtualization
- [HVF](https://developer.apple.com/documentation/hypervisor) - macOS Hypervisor.framework
- [cgroups v2](https://docs.kernel.org/admin-guide/cgroup-v2.html) - Resource isolation
- [seccomp](https://man7.org/linux/man-pages/man2/seccomp.2.html) - Syscall filtering

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_concurrent_vms` | 10 | Maximum parallel VMs (backpressure) |
| `warm_pool_size` | 0 | Enable warm pool (>0). Size = `max_concurrent_vms × 25%` per language |
| `default_memory_mb` | 256 | Guest VM memory (128-2048 MB). Effective ~25% higher with zram |
| `default_timeout_seconds` | 30 | Execution timeout (1-300s) |
| `images_dir` | auto | VM images directory |
| `snapshot_cache_dir` | /tmp/exec-sandbox-cache | Local snapshot cache (L1) |
| `s3_bucket` | None | S3 bucket for snapshots (L3) |
| `s3_region` | us-east-1 | AWS region |
| `enable_package_validation` | True | Validate against top 10k PyPI/npm |

Environment variables: `EXEC_SANDBOX_MAX_CONCURRENT_VMS`, `EXEC_SANDBOX_IMAGES_DIR`, etc.

## Memory Optimization

VMs include automatic memory optimization that requires **no configuration**:

| Feature | What It Does | Benefit |
|---------|--------------|---------|
| **zram** | Compressed swap in RAM (lz4) | +10-30% usable memory |
| **virtio-balloon** | Reclaims unused pages before snapshots | 70-90% smaller snapshots |

**Effective memory (scales with configured RAM):**

| Configured | Available | zram | Effective Capacity |
|------------|-----------|------|-------------------|
| 256MB | ~175MB | 108MB | ~280-320MB |
| 512MB | ~420MB | 233MB | ~600-700MB |
| 1024MB | ~890MB | 484MB | ~1200-1400MB |

zram is always 50% of total RAM, so larger VMs get proportionally more expansion.

**Balloon savings (before snapshots):**

| Configured | Reclaimable | Snapshot Size |
|------------|-------------|---------------|
| 256MB | ~180MB | ~70MB |
| 512MB | ~430MB | ~80MB |
| 1024MB | ~900MB | ~120MB |

Balloon deflates to 64MB minimum, so larger VMs see bigger snapshot reductions.

**What this means for users:**

- Memory-heavy code execution works better than the configured limit suggests
- Compression ratio depends on data: 18-50x for code/text, ~1x for encrypted/random data
- Latency impact is negligible (~0.2μs per memory access)
- Snapshot restore is faster due to smaller files

**Tradeoffs:**

- CPU overhead for compression (minimal with lz4)
- Incompressible data (encryption, media) won't benefit from zram
- OOM errors may be delayed (zram absorbs pressure before failing)

## Execution Result

| Field | Type | Description |
|-------|------|-------------|
| `stdout` | str | Captured output (max 1MB) |
| `stderr` | str | Captured errors (max 100KB) |
| `exit_code` | int | Process exit code (0 = success) |
| `execution_time_ms` | int | Guest-reported duration |
| `external_cpu_time_ms` | int | Host cgroup CPU time |
| `external_memory_peak_mb` | int | Host cgroup peak memory |
| `timing.setup_ms` | int | Resource setup (overlay, cgroup, network) |
| `timing.boot_ms` | int | VM boot (kernel + guest agent) |
| `timing.execute_ms` | int | Code execution |
| `timing.total_ms` | int | End-to-end time |

## Exceptions

| Exception | Description |
|-----------|-------------|
| `SandboxError` | Base exception |
| `SandboxDependencyError` | Optional dependency missing (e.g., aioboto3 for S3) |
| `VmError` | VM operation failed |
| `VmTimeoutError` | Execution exceeded timeout |
| `VmBootError` | VM failed to start |
| `CommunicationError` | Host-guest communication failed |
| `GuestAgentError` | Guest agent returned error |
| `PackageNotAllowedError` | Package not in allowlist |
| `SnapshotError` | Snapshot operation failed |

## Pitfalls

```python
# VMs are never reused - state doesn't persist
result1 = await scheduler.run("x = 42", language="python")
result2 = await scheduler.run("print(x)", language="python")  # NameError!
# Fix: single execution with all code
await scheduler.run("x = 42; print(x)", language="python")

# Warm pool only works without packages
config = SchedulerConfig(warm_pool_size=1)
await scheduler.run(code="...", packages=["pandas"])  # Bypasses warm pool, cold boot (400ms)
await scheduler.run(code="...")                        # Uses warm pool (1-2ms)

# Pin package versions for caching
packages=["pandas==2.2.0"]  # Cacheable
packages=["pandas"]         # Cache miss every time

# Streaming callbacks must be fast (blocks event loop)
on_stdout=lambda chunk: time.sleep(1)        # Blocks!
on_stdout=lambda chunk: buffer.append(chunk)  # Fast

# Memory overhead: warm pool uses (max_concurrent_vms × 25%) × 2 languages × 256MB
# max_concurrent_vms=20 → 5 VMs/lang × 2 × 256MB = 2.5GB for warm pool alone

# Memory can exceed configured limit due to zram compression
default_memory_mb=256  # Code can actually use ~280-320MB thanks to zram
# Don't rely on memory limits for security - use timeouts for runaway allocations

# Network without domain restrictions is risky
allow_network=True                              # Full internet access
allow_network=True, allowed_domains=["api.example.com"]  # Controlled
```

## Limits

| Resource | Limit |
|----------|-------|
| Max code size | 1MB |
| Max stdout | 1MB |
| Max stderr | 100KB |
| Max packages | 50 |
| Max env vars | 100 |
| Execution timeout | 1-300s |
| VM memory | 128-2048MB |
| Max concurrent VMs | 1-100 |

## Security Architecture

| Layer | Technology | Protection |
|-------|------------|------------|
| 1 | KVM/HVF virtualization | Hardware-enforced CPU isolation (ring -1) |
| 2 | Unprivileged QEMU | No root privileges, reduced attack surface |
| 3 | Seccomp | Syscall filtering, deny-by-default |
| 4 | cgroups v2 | Memory, CPU, PID limits (fork bomb prevention) |
| 5 | Linux namespaces | PID, network, mount, UTS, IPC isolation |
| 6 | MAC | AppArmor/SELinux when available |

**Guarantees:**

- VMs are never reused - fresh VM per `run()`, destroyed immediately after
- Network disabled by default - requires explicit `allow_network=True`
- DNS whitelisting - only specified domains accessible when network enabled
- Package validation - only top 10k PyPI/npm packages allowed by default

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| QEMU | 8.0+ |
| Hardware acceleration | KVM (Linux) or HVF (macOS) |

Verify acceleration:

```bash
ls /dev/kvm              # Linux
sysctl kern.hv_support   # macOS
```

Without hardware acceleration, QEMU uses TCG software emulation (10-50x slower).

## VM Images

Pre-built images from [GitHub Releases](https://github.com/dualeai/exec-sandbox/releases):

| Image | Contents | Size |
|-------|----------|------|
| python-3.14-base | Alpine + Python 3.14 + uv + guest-agent | ~512MB |
| node-23-base | Alpine + Node.js 23 + bun + guest-agent | ~512MB |

Build from source:

```bash
./scripts/build-images.sh
# Output: ./images/.dist/python-3.14-base.qcow2, ./images/.dist/node-23-base.qcow2
```

## Security

- [Security Policy](./SECURITY.md) - Vulnerability reporting
- [SBOM](https://github.com/dualeai/exec-sandbox/releases) - Software Bill of Materials (SPDX format) attached to releases

## Contributing

Contributions welcome! Please open an issue first to discuss changes.

```bash
make install      # Setup venv
make test         # Run tests
make lint         # Format and lint
```

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

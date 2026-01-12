# exec-sandbox

Secure code execution in isolated lightweight VMs (QEMU microVMs). Python library for running untrusted Python, JavaScript, and shell code with 7-layer security isolation.

[![CI](https://github.com/dualeai/exec-sandbox/actions/workflows/test.yml/badge.svg)](https://github.com/dualeai/exec-sandbox/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/codecov/c/github/dualeai/exec-sandbox)](https://codecov.io/gh/dualeai/exec-sandbox)
[![PyPI](https://img.shields.io/pypi/v/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![Python](https://img.shields.io/pypi/pyversions/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![License](https://img.shields.io/pypi/l/exec-sandbox)](https://opensource.org/licenses/Apache-2.0)

## Highlights

- **Hardware isolation** - Each execution runs in a dedicated lightweight VM (QEMU with KVM/HVF hardware acceleration), not containers
- **Fast startup** - 400ms fresh start, 1-2ms with pre-started VMs (warm pool)
- **Simple API** - Just `Scheduler` and `run()`, async-friendly
- **Streaming output** - Real-time output as code runs
- **Smart caching** - Local + S3 remote cache for VM snapshots
- **Network control** - Disabled by default, optional domain allowlisting
- **Memory optimization** - Compressed memory (zram) + unused memory reclamation (balloon) for ~30% more capacity, ~80% smaller snapshots

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

VM images are **automatically downloaded** from GitHub Releases on first use. No manual setup required.

## Asset Downloads

exec-sandbox requires VM images (kernel, initramfs, qcow2) and binaries (gvproxy-wrapper) to run. These assets are **automatically downloaded** from GitHub Releases on first use.

### How it works

1. On first `Scheduler` initialization, exec-sandbox checks if assets exist in the cache directory
2. If missing, it queries the GitHub Releases API for the matching version (`v{__version__}`)
3. Assets are downloaded over HTTPS, verified against SHA256 checksums (provided by GitHub API), and decompressed
4. Subsequent runs use the cached assets (no re-download)

### Cache locations

| Platform | Location |
|----------|----------|
| macOS | `~/Library/Caches/exec-sandbox/` |
| Linux | `~/.cache/exec-sandbox/` (or `$XDG_CACHE_HOME/exec-sandbox/`) |

### Environment variables

| Variable | Description |
|----------|-------------|
| `EXEC_SANDBOX_CACHE_DIR` | Override cache directory |
| `EXEC_SANDBOX_OFFLINE` | Set to `1` to disable auto-download (fail if assets missing) |
| `EXEC_SANDBOX_ASSET_VERSION` | Force specific release version |

### Security considerations

**Integrity verification:**
- All downloaded assets are verified against SHA256 checksums
- Checksums are fetched from GitHub's Release API (computed at upload time, immutable)
- If checksum verification fails, the download is rejected and retried

**Supply chain security:**
- Assets are downloaded exclusively over HTTPS from `github.com`
- Release assets are built in GitHub Actions CI with [build provenance attestations](https://docs.github.com/en/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds)
- You can verify attestations with: `gh attestation verify <asset> --owner dualeai`

**Offline/air-gapped environments:**
- Pre-download assets: `gh release download v{version} -R dualeai/exec-sandbox -D ~/.cache/exec-sandbox/`
- Set `EXEC_SANDBOX_OFFLINE=1` to prevent any network requests
- Or point `EXEC_SANDBOX_CACHE_DIR` to a pre-populated directory

**Network access:**
- Only connects to `api.github.com` (release metadata) and `github.com` (asset downloads)
- No telemetry or analytics
- All requests are read-only (GET)

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
        allowed_domains=["httpbin.org"],  # Domain allowlist
    )
```

### Production Configuration

```python
from exec_sandbox import Scheduler, SchedulerConfig

config = SchedulerConfig(
    max_concurrent_vms=20,       # Limit parallel executions
    warm_pool_size=1,            # Pre-started VMs (warm pool), size = max_concurrent_vms × 25%
    default_memory_mb=512,       # Per-VM memory
    default_timeout_seconds=60,  # Execution timeout
    s3_bucket="my-snapshots",    # Remote cache for package snapshots
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
- [KVM](https://www.linux-kvm.org/page/Documents) - Linux hardware virtualization
- [HVF](https://developer.apple.com/documentation/hypervisor) - macOS hardware virtualization (Hypervisor.framework)
- [cgroups v2](https://docs.kernel.org/admin-guide/cgroup-v2.html) - Linux resource limits
- [seccomp](https://man7.org/linux/man-pages/man2/seccomp.2.html) - System call filtering

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_concurrent_vms` | 10 | Maximum parallel VMs |
| `warm_pool_size` | 0 | Pre-started VMs (warm pool). Set >0 to enable. Size = `max_concurrent_vms × 25%` per language |
| `default_memory_mb` | 256 | VM memory (128-2048 MB). Effective ~25% higher with memory compression (zram) |
| `default_timeout_seconds` | 30 | Execution timeout (1-300s) |
| `images_dir` | auto | VM images directory |
| `snapshot_cache_dir` | /tmp/exec-sandbox-cache | Local snapshot cache |
| `s3_bucket` | None | S3 bucket for remote snapshot cache |
| `s3_region` | us-east-1 | AWS region |
| `enable_package_validation` | True | Validate against top 10k packages (PyPI for Python, npm for JavaScript) |
| `auto_download_assets` | True | Auto-download VM images from GitHub Releases |

Environment variables: `EXEC_SANDBOX_MAX_CONCURRENT_VMS`, `EXEC_SANDBOX_IMAGES_DIR`, etc.

## Memory Optimization

VMs include automatic memory optimization that requires **no configuration**:

| Feature | What It Does | Benefit |
|---------|--------------|---------|
| **Compressed swap (zram)** | Uses RAM as compressed swap space (lz4 compression) | +10-30% usable memory |
| **Memory reclamation (virtio-balloon)** | Returns unused memory before snapshots | 70-90% smaller snapshots |

**Effective memory (scales with configured RAM):**

| Configured | Available | Compressed Swap | Effective Capacity |
|------------|-----------|-----------------|-------------------|
| 256MB | ~175MB | 108MB | ~280-320MB |
| 512MB | ~420MB | 233MB | ~600-700MB |
| 1024MB | ~890MB | 484MB | ~1200-1400MB |

Compressed swap (zram) is always 50% of total RAM, so larger VMs get proportionally more expansion.

**Memory reclamation savings (before snapshots):**

| Configured | Reclaimable | Snapshot Size |
|------------|-------------|---------------|
| 256MB | ~180MB | ~70MB |
| 512MB | ~430MB | ~80MB |
| 1024MB | ~900MB | ~120MB |

Memory reclamation (balloon) shrinks to 64MB minimum, so larger VMs see bigger snapshot reductions.

**What this means for users:**

- Memory-heavy code execution works better than the configured limit suggests
- Compression ratio depends on data: 18-50x for code/text, ~1x for encrypted/random data
- Speed impact is negligible (~0.2μs delay per memory access)
- Snapshot restore is faster due to smaller files

**Tradeoffs:**

- CPU overhead for compression (minimal with lz4)
- Incompressible data (encryption, media) won't benefit from compressed swap
- Out-of-memory (OOM) errors may be delayed (compressed swap absorbs pressure before failing)

## Execution Result

| Field | Type | Description |
|-------|------|-------------|
| `stdout` | str | Captured output (max 1MB) |
| `stderr` | str | Captured errors (max 100KB) |
| `exit_code` | int | Process exit code (0 = success) |
| `execution_time_ms` | int | Duration reported by VM |
| `external_cpu_time_ms` | int | CPU time measured by host |
| `external_memory_peak_mb` | int | Peak memory measured by host |
| `timing.setup_ms` | int | Resource setup (filesystem, limits, network) |
| `timing.boot_ms` | int | VM boot time |
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
| `CommunicationError` | VM communication failed |
| `SocketAuthError` | Socket peer authentication failed |
| `GuestAgentError` | VM helper process returned error |
| `PackageNotAllowedError` | Package not in allowlist |
| `SnapshotError` | Snapshot operation failed |
| `AssetError` | Asset download/verification error (base) |
| `AssetDownloadError` | Asset download failed |
| `AssetChecksumError` | Asset checksum verification failed |
| `AssetNotFoundError` | Asset not found in registry/release |

## Pitfalls

```python
# VMs are never reused - state doesn't persist
result1 = await scheduler.run("x = 42", language="python")
result2 = await scheduler.run("print(x)", language="python")  # NameError!
# Fix: single execution with all code
await scheduler.run("x = 42; print(x)", language="python")

# Pre-started VMs (warm pool) only work without packages
config = SchedulerConfig(warm_pool_size=1)
await scheduler.run(code="...", packages=["pandas"])  # Bypasses warm pool, fresh start (400ms)
await scheduler.run(code="...")                        # Uses warm pool (1-2ms)

# Pin package versions for caching
packages=["pandas==2.2.0"]  # Cacheable
packages=["pandas"]         # Cache miss every time

# Streaming callbacks must be fast (blocks async execution)
on_stdout=lambda chunk: time.sleep(1)        # Blocks!
on_stdout=lambda chunk: buffer.append(chunk)  # Fast

# Memory overhead: pre-started VMs use (max_concurrent_vms × 25%) × 2 languages × 256MB
# max_concurrent_vms=20 → 5 VMs/lang × 2 × 256MB = 2.5GB for warm pool alone

# Memory can exceed configured limit due to compressed swap
default_memory_mb=256  # Code can actually use ~280-320MB thanks to compression
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
| 1 | Hardware virtualization (KVM/HVF) | CPU isolation enforced by hardware |
| 2 | Unprivileged QEMU | No root privileges, minimal exposure |
| 3 | System call filtering (seccomp) | Blocks unauthorized OS calls |
| 4 | Resource limits (cgroups v2) | Memory, CPU, process limits |
| 5 | Process isolation (namespaces) | Separate process, network, filesystem views |
| 6 | Security policies (AppArmor/SELinux) | When available |
| 7 | Socket authentication (SO_PEERCRED/LOCAL_PEERCRED) | Verifies QEMU process identity |

**Guarantees:**

- VMs are never reused - fresh VM per `run()`, destroyed immediately after
- Network disabled by default - requires explicit `allow_network=True`
- Domain allowlisting - only specified domains accessible when network enabled
- Package validation - only top 10k Python/JavaScript packages allowed by default

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| QEMU | 8.0+ |
| Hardware acceleration | KVM (Linux) or HVF (macOS) |

Verify hardware acceleration is available:

```bash
ls /dev/kvm              # Linux
sysctl kern.hv_support   # macOS
```

Without hardware acceleration, QEMU uses software emulation (TCG), which is 10-50x slower.

## VM Images

Pre-built images from [GitHub Releases](https://github.com/dualeai/exec-sandbox/releases):

| Image | Contents | Size |
|-------|----------|------|
| python-3.14-base | Alpine Linux + Python 3.14 + uv + VM helper | ~512MB |
| node-23-base | Alpine Linux + Node.js 23 + bun + VM helper | ~512MB |

Build from source:

```bash
./scripts/build-images.sh
# Output: ./images/.dist/python-3.14-base.qcow2, ./images/.dist/node-23-base.qcow2 (QEMU disk images)
```

## Security

- [Security Policy](./SECURITY.md) - Vulnerability reporting
- [Dependency list (SBOM)](https://github.com/dualeai/exec-sandbox/releases) - Full list of included software, attached to releases

## Contributing

Contributions welcome! Please open an issue first to discuss changes.

```bash
make install      # Setup environment
make test         # Run tests
make lint         # Format and lint
```

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

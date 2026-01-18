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
- **Simple API** - Just `Scheduler` and `run()`, async-friendly; plus `sbx` CLI for quick testing
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

## Quick Start

### CLI

The `sbx` command provides quick access to sandbox execution from the terminal:

```bash
# Run Python code
sbx 'print("Hello, World!")'

# Run JavaScript
sbx -l javascript 'console.log("Hello!")'

# Run a file (language auto-detected from extension)
sbx script.py
sbx app.js

# From stdin
echo 'print(42)' | sbx -

# With packages
sbx -p requests -p pandas 'import pandas; print(pandas.__version__)'

# With timeout and memory limits
sbx -t 60 -m 512 long_script.py

# Enable network with domain allowlist
sbx --network --allow-domain api.example.com fetch_data.py

# JSON output for scripting
sbx --json 'print("test")' | jq .exit_code

# Environment variables
sbx -e API_KEY=secret -e DEBUG=1 script.py

# Multiple sources (run concurrently)
sbx 'print(1)' 'print(2)' script.py

# Multiple inline codes
sbx -c 'print(1)' -c 'print(2)'

# Limit concurrency
sbx -j 5 *.py
```

**CLI Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--language` | `-l` | python, javascript, raw | auto-detect |
| `--code` | `-c` | Inline code (repeatable, alternative to positional) | - |
| `--package` | `-p` | Package to install (repeatable) | - |
| `--timeout` | `-t` | Timeout in seconds | 30 |
| `--memory` | `-m` | Memory in MB | 256 |
| `--env` | `-e` | Environment variable KEY=VALUE (repeatable) | - |
| `--network` | | Enable network access | false |
| `--allow-domain` | | Allowed domain (repeatable) | - |
| `--json` | | JSON output | false |
| `--quiet` | `-q` | Suppress progress output | false |
| `--no-validation` | | Skip package allowlist validation | false |
| `--concurrency` | `-j` | Max concurrent VMs for multi-input | 10 |

### Python API

#### Basic Execution

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

#### With Packages

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

#### Streaming Output

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="for i in range(5): print(i)",
        language="python",
        on_stdout=lambda chunk: print(f"[OUT] {chunk}", end=""),
        on_stderr=lambda chunk: print(f"[ERR] {chunk}", end=""),
    )
```

#### Network Access

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="import urllib.request; print(urllib.request.urlopen('https://httpbin.org/ip').read())",
        language="python",
        allow_network=True,
        allowed_domains=["httpbin.org"],  # Domain allowlist
    )
```

#### Production Configuration

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

#### Error Handling

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

### Security

Assets are verified against SHA256 checksums and built with [provenance attestations](https://docs.github.com/en/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds). For offline environments, set `EXEC_SANDBOX_OFFLINE=1` after pre-downloading assets.

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

VMs include automatic memory optimization (no configuration required):

- **Compressed swap (zram)** - ~25% more usable memory via lz4 compression
- **Memory reclamation (virtio-balloon)** - 70-90% smaller snapshots

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

| Requirement | Supported |
|-------------|-----------|
| Python | 3.12, 3.13, 3.14 (including free-threaded) |
| Linux | x64, arm64 |
| macOS | x64, arm64 |
| QEMU | 8.0+ |
| Hardware acceleration | KVM (Linux) or HVF (macOS) recommended, 10-50x faster |

Verify hardware acceleration is available:

```bash
ls /dev/kvm              # Linux
sysctl kern.hv_support   # macOS
```

Without hardware acceleration, QEMU uses software emulation (TCG), which is 10-50x slower.

### Linux Setup (Optional Security Hardening)

For enhanced security on Linux, exec-sandbox can run QEMU as an unprivileged `qemu-vm` user. This isolates the VM process from your user account.

```bash
# Create qemu-vm system user
sudo useradd --system --no-create-home --shell /usr/sbin/nologin qemu-vm

# Add qemu-vm to kvm group (for hardware acceleration)
sudo usermod -aG kvm qemu-vm

# Add your user to qemu-vm group (for socket access)
sudo usermod -aG qemu-vm $USER

# Re-login or activate group membership
newgrp qemu-vm
```

**Why is this needed?** When `qemu-vm` user exists, exec-sandbox runs QEMU as that user for process isolation. The host needs to connect to QEMU's Unix sockets (0660 permissions), which requires group membership. This follows the [libvirt security model](https://wiki.archlinux.org/title/Libvirt).

If `qemu-vm` user doesn't exist, exec-sandbox runs QEMU as your user (no additional setup required, but less isolated).

## VM Images

Pre-built images from [GitHub Releases](https://github.com/dualeai/exec-sandbox/releases):

| Image | Runtime | Package Manager | Size | Description |
|-------|---------|-----------------|------|-------------|
| `python-3.14-base` | Python 3.14 | uv | ~140MB | Full Python environment with C extension support |
| `node-1.3-base` | Bun 1.3 | bun | ~57MB | Fast JavaScript/TypeScript runtime with Node.js compatibility |
| `raw-base` | None | None | ~15MB | Shell scripts and custom runtimes |

All images are based on **Alpine Linux 3.21** (Linux 6.12 LTS, musl libc) and include common tools for AI agent workflows.

### Common Tools (all images)

| Tool | Purpose |
|------|---------|
| `git` | Version control, clone repositories |
| `curl` | HTTP requests, download files |
| `jq` | JSON processing |
| `bash` | Shell scripting |
| `coreutils` | Standard Unix utilities (ls, cp, mv, etc.) |
| `tar`, `gzip`, `unzip` | Archive extraction |
| `file` | File type detection |

### Python Image

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.14 | [python-build-standalone](https://github.com/astral-sh/python-build-standalone) (musl) |
| uv | 0.9+ | 10-100x faster than pip ([docs](https://docs.astral.sh/uv/)) |
| gcc, musl-dev | Alpine | For C extensions (numpy, pandas, etc.) |

**Usage notes:**
- Use `uv pip install` instead of `pip install` (pip not included)
- Python 3.14 includes t-strings, deferred annotations, free-threading support

### JavaScript Image

| Component | Version | Notes |
|-----------|---------|-------|
| Bun | 1.3 | Runtime, bundler, package manager ([docs](https://bun.com/docs)) |

**Usage notes:**
- Bun is a Node.js-compatible runtime (not Node.js itself)
- Built-in TypeScript/JSX support, no transpilation needed
- Use `bun install` for packages, `bun run` for scripts
- Near-complete Node.js API compatibility

### Raw Image

Minimal Alpine Linux with common tools only. Use for:
- Shell script execution (`language="raw"`)
- Custom runtime installation
- Lightweight workloads

Build from source:

```bash
./scripts/build-images.sh
# Output: ./images/dist/python-3.14-base.qcow2, ./images/dist/node-1.3-base.qcow2, ./images/dist/raw-base.qcow2
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

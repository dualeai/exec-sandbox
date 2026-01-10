# exec-sandbox

Secure code execution in microVMs with 6-layer security isolation.

[![PyPI](https://img.shields.io/pypi/v/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Highlights

- **Secure isolation**: 6-layer security architecture (KVM, unprivileged QEMU, seccomp, cgroups, namespaces, MAC)
- **Fast startup**: 400ms cold boot, 1-2ms with warm pool
- **Simple API**: Two primitives - `Scheduler` and `run()`
- **Streaming output**: Real-time stdout/stderr via callbacks
- **Package caching**: L1 (local) + L3 (S3) snapshot cache for fast package installation

## Quick Start

```python
from exec_sandbox import Scheduler

async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="print('Hello, World!')",
        language="python",
    )
    print(result.stdout)  # "Hello, World!\n"
    print(result.exit_code)  # 0
```

## Installation

```bash
# Install package
pip install exec-sandbox

# Install QEMU
brew install qemu          # macOS
apt install qemu-system    # Ubuntu/Debian

# Download VM images (from GitHub Releases)
curl -LO https://github.com/dualeai/exec-sandbox/releases/latest/download/images-x86_64.tar.zst
mkdir -p ~/.local/share/exec-sandbox/images
tar --zstd -xf images-x86_64.tar.zst -C ~/.local/share/exec-sandbox/images/
```

For S3 snapshot caching:

```bash
pip install exec-sandbox[s3]
```

## Examples

### Basic Execution

```python
from exec_sandbox import Scheduler

async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="print(1 + 1)",
        language="python",
    )
    assert result.stdout.strip() == "2"
    assert result.exit_code == 0
```

### With Packages

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="import pandas; print(pandas.__version__)",
        language="python",
        packages=["pandas==2.2.0"],
    )
    print(result.stdout)  # "2.2.0\n"
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
        allowed_domains=["httpbin.org"],  # Whitelist-only access
    )
```

### JavaScript Execution

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="console.log('Hello from Node.js!')",
        language="javascript",
    )
```

### Production Configuration

```python
from exec_sandbox import Scheduler, SchedulerConfig

config = SchedulerConfig(
    max_concurrent_vms=20,
    warm_pool_size=3,  # Pre-boot 3 VMs per language
    default_memory_mb=512,
    default_timeout_seconds=60,
    s3_bucket="my-snapshot-cache",
    s3_region="us-east-1",
)

async with Scheduler(config) as scheduler:
    result = await scheduler.run(...)
```

## Configuration

### SchedulerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent_vms` | int | 10 | Maximum concurrent VMs (backpressure control) |
| `warm_pool_size` | int | 0 | Pre-booted VMs per language (0 disables) |
| `default_memory_mb` | int | 256 | Default guest VM memory (128-2048) |
| `default_timeout_seconds` | int | 30 | Default execution timeout (1-300) |
| `images_dir` | Path | auto | Directory containing VM images |
| `snapshot_cache_dir` | Path | /tmp/exec-sandbox-cache | Local snapshot cache |
| `s3_bucket` | str | None | S3 bucket for snapshot backup |
| `s3_region` | str | us-east-1 | AWS region for S3 |
| `enable_package_validation` | bool | True | Validate packages against allowlist |

### Environment Variables

All config options can be set via environment variables with `EXEC_SANDBOX_` prefix:

```bash
export EXEC_SANDBOX_MAX_CONCURRENT_VMS=20
export EXEC_SANDBOX_IMAGES_DIR=/path/to/images
export EXEC_SANDBOX_S3_BUCKET=my-snapshots
```

## API Reference

### Scheduler

```python
class Scheduler:
    def __init__(self, config: SchedulerConfig | None = None) -> None: ...

    async def __aenter__(self) -> Scheduler: ...
    async def __aexit__(self, *args) -> None: ...

    async def run(
        self,
        code: str,
        *,
        language: Literal["python", "javascript"],
        packages: list[str] | None = None,
        timeout_seconds: int | None = None,
        memory_mb: int | None = None,
        allow_network: bool = False,
        allowed_domains: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> ExecutionResult: ...
```

### ExecutionResult

```python
class ExecutionResult:
    stdout: str                      # Standard output (truncated at 1MB)
    stderr: str                      # Standard error (truncated at 100KB)
    exit_code: int                   # Process exit code (0 = success)
    execution_time_ms: int | None    # Execution time (guest-reported)
    external_cpu_time_ms: int | None # CPU time (host cgroup)
    external_memory_peak_mb: int | None  # Peak memory (host cgroup)
```

### Exceptions

```python
SandboxError                    # Base exception
├── SandboxDependencyError      # Optional dependency missing (e.g., aioboto3)
├── VmError                     # VM operation failed
│   ├── VmTimeoutError          # Execution timeout
│   └── VmBootError             # VM boot failed
├── CommunicationError          # Guest communication failed
├── GuestAgentError             # Guest agent error response
├── PackageNotAllowedError      # Package not in allowlist
└── SnapshotError               # Snapshot operation failed
```

## Security Architecture

exec-sandbox implements 6 layers of defense-in-depth isolation:

| Layer | Technology | Protection |
|-------|------------|------------|
| 1 | KVM/HVF virtualization | CPU ring -1 isolation (hardware-enforced) |
| 2 | Unprivileged QEMU | No root access, reduced attack surface |
| 3 | Seccomp | Syscall filtering (deny-by-default) |
| 4 | cgroups v2 | Memory, CPU, PID limits |
| 5 | Linux namespaces | PID, network, mount, UTS, IPC isolation |
| 6 | MAC (AppArmor/SELinux) | Mandatory access control (when available) |

### VM Lifecycle Security

- VMs are **never reused** between executions (one-time use guarantee)
- Fresh VM per `run()` call, destroyed immediately after
- Warm pool VMs are pre-booted but still one-time use

### Network Isolation

- Default: Network disabled (complete isolation)
- Optional: DNS-based domain whitelist via gvisor-tap-vsock
- No direct internet access without explicit `allow_network=True`

### Package Validation

- Default: Packages validated against top 10k PyPI/npm allowlist
- Prevents installation of unknown/malicious packages
- Can be disabled for testing: `enable_package_validation=False`

## Requirements

- Python 3.12+
- QEMU 8.0+ with KVM (Linux) or HVF (macOS) acceleration
- VM images from GitHub Releases

### Hardware Acceleration

| Platform | Acceleration | Check |
|----------|--------------|-------|
| Linux | KVM | `ls /dev/kvm` |
| macOS | HVF | `sysctl kern.hv_support` |

Without hardware acceleration, QEMU falls back to TCG (software emulation), which is significantly slower.

## Base Images

Zero-maintenance base images for code execution.

### Architecture

```
scripts/build-images.sh
├── Guest Agent Builder (Rust → static musl binary)
├── Python Base Image (Alpine + Python + guest-agent)
├── Node Base Image (Alpine + Node.js + guest-agent)
└── Output: qcow2 images
```

### Building

```bash
# Build all base images
./scripts/build-images.sh

# Result:
# ./images/.dist/python-3.14-base.qcow2
# ./images/.dist/node-23-base.qcow2
```

### What's Included

**Python 3.14 Base Image**

- Alpine Linux 3.21
- Python 3.14 + uv
- Guest agent (virtio-serial listener)
- Minimal size: ~512MB

**Node 23 Base Image**

- Alpine Linux 3.21
- Node.js 23 + bun
- Guest agent (virtio-serial listener)
- Minimal size: ~512MB

### Guest Agent

The guest agent listens on virtio-serial ports:

- `/dev/virtio-ports/org.dualeai.cmd` (host → guest)
- `/dev/virtio-ports/org.dualeai.event` (guest → host)

Commands:

- `ping` - Health check
- `install_packages` - Install pip/npm packages
- `exec` - Run code with timeout

### Guest Agent Development

```bash
cd guest-agent
cargo build --release
./target/release/guest-agent
```

### Integration with SnapshotManager

```python
# SnapshotManager uses base images automatically
snapshot_manager = SnapshotManager(settings)

# First request: boots base image, installs packages, creates snapshot
snapshot = await snapshot_manager.get_or_create_snapshot(
    language="python",
    packages=["pandas==2.1.0", "numpy==1.26.0"],
)

# Subsequent requests: <150ms restore from snapshot
```

## Development

```bash
# Clone and install
git clone https://github.com/dualeai/exec-sandbox.git
cd exec-sandbox
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check .
uv run pyright .
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

# exec-sandbox

Run untrusted code safely in isolated microVMs.

<!-- Status -->
[![Build](https://img.shields.io/github/actions/workflow/status/dualeai/exec-sandbox/ci.yml?branch=main)](https://github.com/dualeai/exec-sandbox/actions)
[![Coverage](https://img.shields.io/codecov/c/github/dualeai/exec-sandbox)](https://codecov.io/gh/dualeai/exec-sandbox)

<!-- Package -->
[![PyPI](https://img.shields.io/pypi/v/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![Python](https://img.shields.io/pypi/pyversions/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Highlights

- **Hardware-level isolation** — Each execution runs in a dedicated QEMU microVM with KVM/HVF acceleration, not just a container
- **Fast cold boot** — 400ms to first code execution, 1-2ms with warm pool pre-booting
- **Two-method API** — Just `Scheduler` and `run()`, no complex configuration required
- **Streaming output** — Real-time stdout/stderr via callbacks for long-running code
- **Smart caching** — L1 local + L3 S3 snapshot cache eliminates repeated package installs

## Installation

```bash
pip install exec-sandbox
```

```bash
# Install QEMU runtime
brew install qemu          # macOS
apt install qemu-system    # Ubuntu/Debian
```

```bash
# Download pre-built VM images
curl -LO https://github.com/dualeai/exec-sandbox/releases/latest/download/images-x86_64.tar.zst
mkdir -p ~/.local/share/exec-sandbox/images
tar --zstd -xf images-x86_64.tar.zst -C ~/.local/share/exec-sandbox/images/
```

For S3 snapshot caching, install with extras:

```bash
pip install exec-sandbox[s3]
```

## Quick Start

```python
from exec_sandbox import Scheduler

async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="print('Hello, World!')",
        language="python",  # or Language.PYTHON
    )
    print(result.stdout)
    # → Hello, World!
    print(result.exit_code)
    # → 0
```

## Examples

### With Packages

Install packages on-demand. First run creates a cached snapshot; subsequent runs restore in <400ms.

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="import pandas; print(pandas.__version__)",
        language="python",
        packages=["pandas==2.2.0"],
    )
    print(result.stdout)
    # → 2.2.0
```

### Streaming Output

Get stdout/stderr in real-time for long-running executions.

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="for i in range(5): print(i)",
        language="python",
        on_stdout=lambda chunk: print(f"[OUT] {chunk}", end=""),
        on_stderr=lambda chunk: print(f"[ERR] {chunk}", end=""),
    )
# → [OUT] 0
# → [OUT] 1
# → [OUT] 2
# → [OUT] 3
# → [OUT] 4
```

### Network Access

Network is disabled by default. Enable with domain whitelisting for controlled access.

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="""
import urllib.request
print(urllib.request.urlopen('https://httpbin.org/ip').read().decode())
""",
        language="python",
        allow_network=True,
        allowed_domains=["httpbin.org"],
    )
    print(result.stdout)
    # → {"origin": "203.0.113.42"}
```

### JavaScript

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="console.log('Hello from Node.js!')",
        language="javascript",
    )
    print(result.stdout)
    # → Hello from Node.js!
```

### Shell Commands

Use `language="raw"` for direct shell execution.

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="echo 'Hello from shell!' && uname -a",
        language="raw",
    )
    print(result.stdout)
    # → Hello from shell!
    # → Linux vm-abc123 6.x.x ...
```

### Error Handling

```python
from exec_sandbox import (
    Scheduler,
    VmTimeoutError,
    PackageNotAllowedError,
    GuestAgentError,
    SandboxError,
)

async with Scheduler() as scheduler:
    try:
        result = await scheduler.run(
            code="while True: pass",
            language="python",
            timeout_seconds=5,
        )
    except VmTimeoutError:
        print("Execution timed out")
    except PackageNotAllowedError as e:
        print(f"Package not in allowlist: {e}")
    except GuestAgentError as e:
        print(f"Execution failed: {e}")
    except SandboxError as e:
        print(f"Sandbox error: {e}")
```

### Timing Breakdown

Cold boots include detailed timing information.

```python
async with Scheduler() as scheduler:
    result = await scheduler.run(
        code="print('hello')",
        language="python",
    )
    if result.timing:
        print(f"Setup: {result.timing.setup_ms}ms")    # Overlay + cgroup + network
        print(f"Boot: {result.timing.boot_ms}ms")      # Kernel + guest agent ready
        print(f"Execute: {result.timing.execute_ms}ms") # Code execution
        print(f"Total: {result.timing.total_ms}ms")    # End-to-end
        # → Setup: 45ms
        # → Boot: 320ms
        # → Execute: 35ms
        # → Total: 400ms
```

### Production Configuration

```python
from exec_sandbox import Scheduler, SchedulerConfig

config = SchedulerConfig(
    max_concurrent_vms=20,       # Handle 20 parallel executions
    warm_pool_size=1,            # Enable warm pool (size = 20 × 25% = 5 per language)
    default_memory_mb=512,       # More memory for data-heavy workloads
    default_timeout_seconds=60,  # Longer timeout for complex computations
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
| `max_concurrent_vms` | int | 10 | Maximum parallel VMs (backpressure control) |
| `warm_pool_size` | int | 0 | Enable warm pool (0 = disabled, >0 = enabled). Actual pool size is 25% of `max_concurrent_vms` per language. |
| `default_memory_mb` | int | 256 | Guest VM memory, 128-2048 MB |
| `default_timeout_seconds` | int | 30 | Execution timeout, 1-300 seconds |
| `images_dir` | Path | auto | Directory containing VM images |
| `snapshot_cache_dir` | Path | /tmp/exec-sandbox-cache | Local snapshot cache (L1) |
| `s3_bucket` | str | None | S3 bucket for snapshot backup (L3) |
| `s3_region` | str | us-east-1 | AWS region for S3 |
| `enable_package_validation` | bool | True | Validate packages against allowlist |

### Environment Variables

All options can be set via environment with `EXEC_SANDBOX_` prefix:

```bash
export EXEC_SANDBOX_MAX_CONCURRENT_VMS=20
export EXEC_SANDBOX_IMAGES_DIR=/path/to/images
export EXEC_SANDBOX_S3_BUCKET=my-snapshots
```

## Best Practices

### Use the Context Manager

The async context manager ensures proper cleanup of VMs and resources.

```python
# Correct - resources properly managed
async with Scheduler() as scheduler:
    result = await scheduler.run(...)

# Incorrect - potential resource leaks
scheduler = Scheduler()
await scheduler.__aenter__()
# forgot __aexit__, VMs may be orphaned
```

### Pin Package Versions

Pinned versions enable snapshot caching. Unpinned versions cause cache misses.

```python
# Correct - reproducible, cacheable
packages=["pandas==2.2.0", "numpy==1.26.0"]

# Incorrect - cache miss every time
packages=["pandas", "numpy>=1.0"]
```

### Size `max_concurrent_vms` to Available Memory

Each VM uses 256-512MB. Size your pool accordingly.

```python
# 32GB host with 50% for VMs = ~30 VMs max
config = SchedulerConfig(max_concurrent_vms=30)
```

### Enable S3 Cache for Production

Local cache is lost on restart. S3 provides persistent snapshot storage.

```python
config = SchedulerConfig(
    s3_bucket="my-snapshots",
    s3_region="us-east-1",
)
```

### Whitelist Network Domains

When enabling network, always specify allowed domains.

```python
# Correct - controlled access
await scheduler.run(
    code=code,
    language="python",
    allow_network=True,
    allowed_domains=["api.example.com", "pypi.org"],
)

# Risky - full internet access
await scheduler.run(
    code=code,
    language="python",
    allow_network=True,  # No domain restrictions
)
```

## Common Gotchas

### VMs Are Never Reused

Each `run()` gets a fresh VM. State does not persist between calls.

```python
# This won't work - x is undefined in second call
result1 = await scheduler.run("x = 42", language="python")
result2 = await scheduler.run("print(x)", language="python")  # NameError!

# Correct - single execution with all code
await scheduler.run("x = 42; print(x)", language="python")
```

### Warm Pool Only Works Without Packages

The warm pool provides 1-2ms allocation, but only for executions with `packages=[]`.

```python
config = SchedulerConfig(max_concurrent_vms=10, warm_pool_size=1)  # Enables pool

# Uses warm pool (1-2ms)
await scheduler.run(code="print(1)", language="python")

# Bypasses warm pool, cold boot (400ms)
await scheduler.run(code="...", language="python", packages=["pandas"])
```

For package-heavy workloads, rely on snapshot caching instead:

```python
config = SchedulerConfig(
    warm_pool_size=0,         # Disable (not useful with packages)
    s3_bucket="my-snapshots", # Enable L3 cache
)
```

### Streaming Callbacks Must Be Fast

Slow callbacks block the event loop.

```python
# Incorrect - blocks event loop
await scheduler.run(
    code=code,
    language="python",
    on_stdout=lambda chunk: time.sleep(1),  # Blocks!
)

# Correct - non-blocking
buffer = []
await scheduler.run(
    code=code,
    language="python",
    on_stdout=lambda chunk: buffer.append(chunk),
)
```

### Memory Adds Up Quickly

Warm pool memory: `(max_concurrent_vms × 25%) × 2 languages × 256MB`

```python
# max_concurrent_vms=20, warm pool enabled:
# pool_size = 20 × 0.25 = 5 VMs per language
# total = 5 × 2 languages × 256MB = 2.5GB just for warm pool
config = SchedulerConfig(max_concurrent_vms=20, warm_pool_size=1)
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
        language: Language,  # "python", "javascript", or "raw"
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
    stdout: str                          # Captured output (max 1MB)
    stderr: str                          # Captured errors (max 100KB)
    exit_code: int                       # 0 = success
    execution_time_ms: int | None        # Guest-reported duration
    external_cpu_time_ms: int | None     # Host cgroup CPU time
    external_memory_peak_mb: int | None  # Host cgroup peak memory
    timing: TimingBreakdown | None       # Detailed timing (cold boot only)
```

### TimingBreakdown

```python
class TimingBreakdown:
    setup_ms: int   # Resource setup (overlay, cgroup, network)
    boot_ms: int    # VM boot (kernel + guest agent ready)
    execute_ms: int # Code execution
    total_ms: int   # End-to-end time
```

### Exceptions

```python
SandboxError                    # Base exception
├── SandboxDependencyError      # Optional dependency missing (e.g., aioboto3)
├── VmError                     # VM operation failed
│   ├── VmTimeoutError          # Execution exceeded timeout
│   └── VmBootError             # VM failed to start
├── CommunicationError          # Host-guest communication failed
├── GuestAgentError             # Guest agent returned error
├── PackageNotAllowedError      # Package not in allowlist
└── SnapshotError               # Snapshot operation failed
```

## Security

### Reporting Vulnerabilities

Report security issues privately via [GitHub Security Advisories](https://github.com/dualeai/exec-sandbox/security/advisories/new). Do not open public issues for security vulnerabilities.

### Supply Chain

- [SECURITY.md](./SECURITY.md) — Vulnerability reporting and disclosure process

### Architecture

exec-sandbox implements 6 layers of defense-in-depth:

| Layer | Technology | Protection |
|-------|------------|------------|
| 1 | KVM/HVF virtualization | Hardware-enforced CPU isolation (ring -1) |
| 2 | Unprivileged QEMU | No root privileges, reduced attack surface |
| 3 | Seccomp | Syscall filtering with deny-by-default policy |
| 4 | cgroups v2 | Memory, CPU, PID limits (fork bomb prevention) |
| 5 | Linux namespaces | PID, network, mount, UTS, IPC isolation |
| 6 | MAC | AppArmor/SELinux when available |

### Isolation Guarantees

- **VMs are never reused** — Fresh VM per `run()` call, destroyed immediately after
- **Network disabled by default** — No internet access without explicit `allow_network=True`
- **DNS whitelisting** — Only specified domains accessible when network enabled
- **Package validation** — Only top 10k PyPI/npm packages allowed by default

## Requirements

- Python 3.12+
- QEMU 8.0+ with hardware acceleration
- VM images from [GitHub Releases](https://github.com/dualeai/exec-sandbox/releases)

### Hardware Acceleration

| Platform | Technology | Verification |
|----------|------------|--------------|
| Linux | KVM | `ls /dev/kvm` |
| macOS | HVF | `sysctl kern.hv_support` |

Without hardware acceleration, QEMU uses software emulation (TCG), which is 10-50x slower.

## Contributing

**Quick links:**

- [Open issues](https://github.com/dualeai/exec-sandbox/issues)
- [Pull requests](https://github.com/dualeai/exec-sandbox/pulls)
- [Discussions](https://github.com/dualeai/exec-sandbox/discussions)

<details>
<summary><strong>Development Setup</strong></summary>

### Clone and Install

```bash
git clone https://github.com/dualeai/exec-sandbox.git
cd exec-sandbox
make install
```

### Run Tests

```bash
make test          # All tests (static + functional)
make test-unit     # Fast unit tests only
make test-e2e      # E2E tests (requires QEMU + images)
```

### Lint and Format

```bash
make lint          # Auto-fix formatting
make test-static   # Check without fixing
```

### Build VM Images

```bash
./scripts/build-images.sh

# Output:
# ./images/.dist/python-3.14-base.qcow2
# ./images/.dist/node-23-base.qcow2
```

</details>

<details>
<summary><strong>VM Image Architecture</strong></summary>

### Build Pipeline

```text
scripts/build-images.sh
├── Guest Agent Builder (Rust → static musl binary)
├── Python Base Image (Alpine + Python 3.14 + uv + guest-agent)
├── Node Base Image (Alpine + Node.js 23 + bun + guest-agent)
└── Output: qcow2 images (~512MB each)
```

### Guest Agent

The guest agent runs inside the VM and handles host commands via virtio-serial:

| Port | Direction | Purpose |
|------|-----------|---------|
| `/dev/virtio-ports/org.dualeai.cmd` | host → guest | Commands |
| `/dev/virtio-ports/org.dualeai.event` | guest → host | Responses |

Commands: `ping`, `install_packages`, `exec`

### Snapshot Integration

```python
# First request: boots base image, installs packages, creates snapshot
# Subsequent requests: <400ms restore from snapshot
snapshot = await snapshot_manager.get_or_create_snapshot(
    language="python",
    packages=["pandas==2.1.0", "numpy==1.26.0"],
)
```

</details>

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

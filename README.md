# exec-sandbox

Secure code execution in isolated lightweight VMs (QEMU microVMs). Python library for running untrusted Python, JavaScript, and shell code with 9-layer security isolation.

[![CI](https://github.com/dualeai/exec-sandbox/actions/workflows/test.yml/badge.svg)](https://github.com/dualeai/exec-sandbox/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/codecov/c/github/dualeai/exec-sandbox)](https://codecov.io/gh/dualeai/exec-sandbox)
[![PyPI](https://img.shields.io/pypi/v/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![Python](https://img.shields.io/pypi/pyversions/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![License](https://img.shields.io/pypi/l/exec-sandbox)](https://opensource.org/licenses/Apache-2.0)

## Highlights

- **Hardware isolation** - Each execution runs in a dedicated lightweight VM (QEMU with KVM/HVF hardware acceleration), not containers
- **Fast startup** - 1-2ms with warm pool, ~100ms from L1 memory snapshot, ~400ms VM boot on cold start (+ 4-11s interpreter startup, amortized by L1)
- **Simple API** - `run()` for one-shot execution, `session()` for stateful multi-step workflows with file I/O; plus `sbx` CLI for quick testing
- **Streaming output** - Real-time output as code runs
- **3-tier snapshot cache** - L1 memory snapshots (~100ms, interpreter warm), L2 disk snapshots (skip package install, interpreter cold-starts), L3 S3 remote cache for cross-host sharing
- **Network control** - Disabled by default, optional domain allowlisting with defense-in-depth filtering (DNS + TLS SNI + DNS cross-validation to prevent spoofing)
- **Memory optimization** - Compressed swap (zram) + unused memory reclamation (virtio-balloon) on idle warm-pool VMs

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
sbx run 'print("Hello from sandbox")'

# Run JavaScript
sbx run -l javascript 'console.log("Hello from sandbox")'

# Run a file (language auto-detected from extension)
sbx run script.py
sbx run app.js

# From stdin
echo 'print(42)' | sbx run -

# With packages
sbx run -p requests -p pandas 'import pandas; print(pandas.__version__)'

# With timeout and memory limits
sbx run -t 60 -m 512 long_script.py

# Enable network with domain allowlist
sbx run --network --allow-domain api.example.com fetch_data.py

# Expose ports (guest:8080 → host:dynamic)
sbx run --expose 8080 --json 'print("ready")' | jq '.exposed_ports[0].url'

# Expose with explicit host port (guest:8080 → host:3000)
sbx run --expose 8080:3000 --json 'print("ready")' | jq '.exposed_ports[0].external'

# Start HTTP server with port forwarding (runs until timeout)
sbx run -t 60 --expose 8080 'import http.server; http.server.test(port=8080, bind="0.0.0.0")'

# JSON output for scripting
sbx run --json 'print("test")' | jq .exit_code

# Environment variables
sbx run -e API_KEY=secret -e DEBUG=1 script.py

# Multiple sources (run concurrently)
sbx run 'print(1)' 'print(2)' script.py

# Multiple inline codes
sbx run -c 'print(1)' -c 'print(2)'

# Debug boot (stream kernel/init logs to stderr)
sbx run --debug 'print("hello")'

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
| `--expose` | | Expose port `INTERNAL[:EXTERNAL][/PROTOCOL]` (repeatable) | - |
| `--json` | | JSON output | false |
| `--quiet` | `-q` | Suppress progress output | false |
| `--no-validation` | | Skip package allowlist validation | false |
| `--upload` | | Upload file `LOCAL:GUEST` (repeatable) | - |
| `--download` | | Download file `GUEST:LOCAL` or `GUEST` (repeatable) | - |
| `--debug` | | Stream kernel/init boot logs to stderr | false |

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

#### Sessions (Stateful Multi-Step)

Sessions keep a VM alive across multiple `exec()` calls — variables, imports, and state persist.

```python
from exec_sandbox import Scheduler

async with Scheduler() as scheduler:
    async with await scheduler.session(language="python") as session:
        await session.exec("import math")
        await session.exec("x = math.pi * 2")
        result = await session.exec("print(f'{x:.4f}')")
        print(result.stdout)  # 6.2832
        print(session.exec_count)  # 3
```

Sessions support all three languages:

```python
# JavaScript/TypeScript — variables and functions persist
async with await scheduler.session(language="javascript") as session:
    await session.exec("const greet = (name: string): string => `Hello, ${name}!`")
    result = await session.exec("console.log(greet('World'))")

# Shell (Bash) — env vars, cwd, and functions persist
async with await scheduler.session(language="raw") as session:
    await session.exec("cd /tmp && export MY_VAR=hello")
    result = await session.exec("echo $MY_VAR from $(pwd)")
```

Sessions auto-close after idle timeout (default: 300s, configurable via `session_idle_timeout_seconds`).

#### File I/O

Sessions support reading, writing, and listing files inside the sandbox.

```python
from pathlib import Path
from exec_sandbox import Scheduler

async with Scheduler() as scheduler:
    async with await scheduler.session(language="python") as session:
        # Write a file into the sandbox
        await session.write_file("input.csv", b"name,score\nAlice,95\nBob,87")

        # Write from a local file
        await session.write_file("model.pkl", Path("./local_model.pkl"))

        # Execute code that reads input and writes output
        await session.exec("data = open('input.csv').read().upper()")
        await session.exec("open('output.csv', 'w').write(data)")

        # Read a file back from the sandbox
        await session.read_file("output.csv", destination=Path("./output.csv"))

        # List files in a directory
        files = await session.list_files("")  # sandbox root
        for f in files:
            print(f"{f.name} {'dir' if f.is_dir else f'{f.size}B'}")
```

CLI file I/O uses sessions under the hood:

```bash
# Upload a local file, run code, download the result
sbx run --upload ./local.csv:input.csv --download output.csv:./result.csv \
  -c "open('output.csv','w').write(open('input.csv').read().upper())"

# Download to ./output.csv (shorthand, no local path)
sbx run --download output.csv -c "open('output.csv','w').write('data')"
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

#### Boot Log Streaming

Stream kernel, tiny-init, and guest-agent boot output for diagnostics:

```python
async with Scheduler() as scheduler:
    boot_lines: list[str] = []
    result = await scheduler.run(
        code="print('hello')",
        language="python",
        on_boot_log=boot_lines.append,  # Automatically enables verbose boot
    )
    for line in boot_lines:
        print(f"[boot] {line}")
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

#### Port Forwarding

Expose VM ports to the host for health checks, API testing, or service validation.

```python
from exec_sandbox import Scheduler, PortMapping

async with Scheduler() as scheduler:
    # Port forwarding without internet (isolated)
    result = await scheduler.run(
        code="print('server ready')",
        expose_ports=[PortMapping(internal=8080, external=3000)],  # Guest:8080 → Host:3000
        allow_network=False,  # No outbound internet
    )
    print(result.exposed_ports[0].url)  # http://127.0.0.1:3000

    # Dynamic port allocation (OS assigns external port)
    result = await scheduler.run(
        code="print('server ready')",
        expose_ports=[8080],  # external=None → OS assigns port
    )
    print(result.exposed_ports[0].external)  # e.g., 52341

    # Long-running server with port forwarding
    result = await scheduler.run(
        code="import http.server; http.server.test(port=8080, bind='0.0.0.0')",
        expose_ports=[PortMapping(internal=8080)],
        timeout_seconds=60,  # Server runs until timeout
    )
```

**Security:** Port forwarding works independently of internet access. When `allow_network=False`, guest VMs cannot initiate outbound connections (all outbound TCP/UDP blocked), but host-to-guest port forwarding still works.

#### Production Configuration

```python
from exec_sandbox import Scheduler, SchedulerConfig

config = SchedulerConfig(
    warm_pool_size=1,            # Pre-started VMs per language (0 disables)
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
from exec_sandbox import (
    Scheduler,
    InputValidationError,
    PackageNotAllowedError,
    SandboxError,
    VmTimeoutError,
)

async with Scheduler() as scheduler:
    try:
        result = await scheduler.run(code="print('hello')", language="python", timeout_seconds=5)
    except InputValidationError as e:
        # Caller bug — bad code or env vars. Fix input and retry.
        # (CodeValidationError, EnvVarValidationError inherit from this)
        print(f"Invalid input: {e}")
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
| `EXEC_SANDBOX_LOG_LEVEL` | Set log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Pre-downloading for offline use

Use `sbx prefetch` to download all assets ahead of time:

```bash
sbx prefetch                    # Download all assets for current arch
sbx prefetch --arch aarch64     # Cross-arch prefetch
sbx prefetch -q                 # Quiet mode (CI/Docker)
```

**Dockerfile example:**

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm
RUN uv pip install --system exec-sandbox
RUN sbx prefetch -q
ENV EXEC_SANDBOX_OFFLINE=1
# Assets cached, no network needed at runtime
```

### Security

Assets are verified against SHA256 checksums and built with [provenance attestations](https://docs.github.com/en/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds).

## Documentation

- [QEMU Documentation](https://www.qemu.org/docs/master/) - Virtual machine emulator
- [KVM](https://www.linux-kvm.org/page/Documents) - Linux hardware virtualization
- [HVF](https://developer.apple.com/documentation/hypervisor) - macOS hardware virtualization (Hypervisor.framework)
- [cgroups v2](https://docs.kernel.org/admin-guide/cgroup-v2.html) - Linux resource limits
- [seccomp](https://man7.org/linux/man-pages/man2/seccomp.2.html) - System call filtering

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warm_pool_size` | 0 | Pre-started VMs per language (Python, JavaScript, Raw). Set >0 to enable |
| `default_memory_mb` | 256 | VM memory (128 MB minimum, no upper bound). Effective ~25% higher with memory compression (zram) |
| `default_timeout_seconds` | 30 | Execution timeout (1-300s) |
| `session_idle_timeout_seconds` | 300 | Session idle timeout (10-3600s). Auto-closes inactive sessions |
| `images_dir` | auto | VM images directory |
| `snapshot_cache_dir` | OS cache dir (see below) | Local snapshot cache (macOS: ~/Library/Caches/exec-sandbox/, Linux: ~/.cache/exec-sandbox/) |
| `s3_bucket` | None | S3 bucket for remote snapshot cache |
| `s3_region` | us-east-1 | AWS region |
| `s3_prefix` | snapshots/ | Prefix for S3 keys |
| `max_concurrent_s3_uploads` | 4 | Max concurrent background S3 uploads (1-16) |
| `memory_overcommit_ratio` | 1.5 | Memory overcommit ratio. Budget = host_total × (1 - reserve) × ratio |
| `cpu_overcommit_ratio` | 4.0 | CPU overcommit ratio. Budget = host_cpus × ratio |
| `host_memory_reserve_ratio` | 0.1 | Fraction of host memory reserved for OS (e.g., 0.1 = 10%) |
| `resource_monitor_interval_seconds` | 5.0 | Interval between resource monitor ticks (1-60s) |
| `enable_package_validation` | True | Validate against top 10k packages (PyPI for Python, npm for JavaScript) |
| `auto_download_assets` | True | Auto-download VM images from GitHub Releases |

Environment variables: `EXEC_SANDBOX_IMAGES_DIR`, `EXEC_SANDBOX_CACHE_DIR`, `EXEC_SANDBOX_OFFLINE`, etc.

## Memory Optimization

VMs include automatic memory optimization (no configuration required):

- **Compressed swap (zram)** - ~25% more usable memory via lz4 compression
- **Memory reclamation (virtio-balloon)** - Reclaims unused guest pages on idle warm-pool VMs (192→160 MB default), reducing host memory pressure

### Memory Architecture

Guest RAM is a fixed budget shared between the kernel, userspace processes, and tmpfs mounts. tmpfs is demand-allocated — writing 10 MB of files consumes ~10 MB of the VM's memory budget. All tmpfs mounts enforce per-UID quota (`usrquota_block_hardlimit`) to prevent sparse file inflation attacks.

```
Guest RAM (default 192 MB)
├── Kernel + slab caches     (~20 MB fixed)
├── Userspace (code execution) (variable)
├── tmpfs mounts (on demand, per-UID quota)
│   ├── /home/user           50% of RAM — user files, packages
│   ├── /tmp                 50% of RAM — pip/uv wheel builds, temp files
│   └── /dev/shm             50% of RAM — POSIX shared memory
└── zram compressed swap     (~25% effective bonus)
```

| Mount | Size | Purpose |
|---|---|---|
| `/home/user` | 50% of RAM | Writable home dir — installed packages, user scripts, data files |
| `/tmp` | 50% of RAM | Scratch space for package managers (wheel builds), temp files |
| `/dev/shm` | 50% of RAM | POSIX shared memory segments (Python multiprocessing semaphores) |

## Snapshot Caching Architecture

exec-sandbox uses a 3-tier snapshot cache to eliminate redundant work across executions. The first run with a given configuration pays the full cost (boot + package install); every subsequent run restores from cache.

```
Request arrives
│
├─ Warm pool hit? ──────────────── 1-2ms     (pre-started VM, REPL already warm)
│
├─ L1 memory snapshot? ────────── ~100ms     (full VM state restore — REPL already warm)
│
├─ L2 disk snapshot? ──────────── ~400ms boot + 4-11s REPL startup
│                                             (packages cached, but interpreter cold-starts)
│
├─ L3 S3 remote cache? ────────── download + same as L2
│
└─ Cold miss ──────────────────── ~400ms boot + 4-11s REPL + package install
```

### L1: Memory Snapshots

L1 captures the complete VM state — CPU registers, RAM pages, device state — via QEMU's [migration subsystem](https://www.qemu.org/docs/master/devel/migration/main.html). Restoring from L1 resumes the VM exactly where it was, with REPL already warm and packages loaded. No boot, no initialization. The guest-agent forces an immediate kernel CRNG reseed (`RNDRESEEDCRNG` ioctl) before every command dispatch, ensuring each restored VM produces unique cryptographic random output despite sharing the same snapshot origin.

On QEMU >= 9.0, exec-sandbox enables [mapped-ram](https://www.qemu.org/docs/master/devel/migration/mapped-ram.html) for fixed-offset page storage (enabling parallel I/O) and [multifd](https://www.qemu.org/docs/master/devel/migration/main.html) for multi-threaded migration channels. Mapped-ram files are sparse — unused RAM pages become filesystem holes. A default 192 MB VM produces a file that appears as ~326 MB (`ls -l`) but consumes only ~50-100 MB of actual disk (Python ~86 MB, JavaScript ~96 MB, raw ~56 MB).

**How L1 saves work:** After a cold boot, exec-sandbox schedules a background save — it boots a sacrificial VM with the same parameters, warms the REPL, saves the migration stream to disk via QMP (`stop` → `migrate` → poll → `quit`), and destroys the VM. The next request with matching parameters restores from the saved vmstate instead of booting.

L1 cache keys include: language, packages, exec-sandbox version, base image hash, QEMU version, CPU architecture, acceleration type, memory size, CPU cores, network topology, migration format version, kernel hash, and initramfs hash. Any parameter change produces a different cache entry.

### L2: Disk Snapshots

L2 stores standalone ext4 qcow2 images with packages pre-installed. When a request needs packages (e.g., `pandas==2.2.0`), exec-sandbox checks L2 for a matching snapshot. On hit, the VM boots with the cached disk overlaid on the read-only EROFS base via overlayfs — skipping package installation but still going through kernel boot (~400ms) and interpreter startup (4-11s for Python/Bun on HVF, ~4-5s on KVM). The snapshot only contains the writable layer (installed packages and their files), not the full rootfs.

This is where L1 matters: L1 snapshots the memory state *after* the interpreter is fully loaded, eliminating the 4-11s REPL startup cost entirely. An L1 save is automatically scheduled after every L2 cold boot, so the interpreter startup penalty is paid only once per unique configuration.

### L3: S3 Remote Cache

L3 extends L2 across machines. When an L2 snapshot is created, it's compressed with zstd and uploaded to S3 in the background. Other hosts can download and decompress it to populate their local L2 cache, avoiding redundant package installations across a fleet.

### How It Compares

The critical metric is **time to first code execution** — not just VM boot, but having the interpreter loaded and ready to run user code.

| Platform | Isolation | Snapshot includes interpreter? | Time to first exec | macOS | License |
|---|---|---|---|---|---|
| **exec-sandbox** (warm pool) | QEMU (KVM/HVF) | Yes (REPL pre-warmed) | 1-2ms | Yes | Apache-2.0 |
| **exec-sandbox** (L1) | QEMU (KVM/HVF) | Yes (memory snapshot) | ~100ms | Yes | Apache-2.0 |
| **exec-sandbox** (L2/L3) | QEMU (KVM/HVF) | No — interpreter cold-starts | ~400ms boot + 4-11s REPL | Yes | Apache-2.0 |
| [**E2B**](https://e2b.dev) | Firecracker (KVM) | [Yes](https://e2b.dev/docs/sandbox/persistence) (VM snapshot) | [~80-150ms](https://e2b.dev/docs) | No | Apache-2.0 |
| [**Lambda SnapStart**](https://docs.aws.amazon.com/lambda/latest/dg/snapstart.html) | Firecracker (KVM) | [Yes](https://brooker.co.za/blog/2022/11/29/snapstart.html) (post-init snapshot) | [~182ms P50](https://aws.amazon.com/blogs/compute/optimizing-cold-start-performance-of-aws-lambda-using-advanced-priming-strategies-with-snapstart/) (Python) | No | AWS service |
| [**Modal**](https://modal.com) | gVisor (userspace kernel) | [Yes](https://modal.com/blog/mem-snapshots) (alpha for sandboxes) | [1-3.5s](https://modal.com/blog/mem-snapshots) | No | Proprietary |
| [**CodeSandbox**](https://codesandbox.io) | Firecracker (KVM) | [Yes](https://codesandbox.io/blog/how-we-clone-a-running-vm-in-2-seconds) (VM snapshot) | [~495ms avg](https://codesandbox.io/blog/how-we-scale-our-microvm-infrastructure-using-low-latency-memory-decompression) | No | Proprietary |
| [**Daytona**](https://daytona.io) | Docker / Kata / Sysbox | [No](https://github.com/daytonaio/daytona/issues/2519) (filesystem only) | Container start + interpreter | No | AGPL-3.0 |

**exec-sandbox's weakness:** L2/L3 hits pay the full interpreter startup cost (4-11s for Python/Bun) on every boot — only L1 and warm pool avoid this. Firecracker-based platforms (E2B, Lambda) always snapshot with the interpreter warm, so they never have this penalty. QEMU is also heavier than Firecracker (~5 MiB VMM overhead for Firecracker).

**exec-sandbox's strength:** Runs on macOS (HVF) and Linux (KVM) with one codebase. Native REPL sessions with state persistence across `exec()` calls. L1 is automatically populated after every L2 cold boot, so the interpreter startup penalty is a one-time cost per unique configuration.

## Execution Result

| Field | Type | Description |
|-------|------|-------------|
| `stdout` | str | Captured output (max 1MB) |
| `stderr` | str | Captured errors (max 100KB) |
| `exit_code` | int | Process exit code (0 = success, 128+N = killed by signal N) |
| `execution_time_ms` | int | Duration reported by VM |
| `external_cpu_time_ms` | int | CPU time measured by host |
| `external_memory_peak_mb` | int | Peak memory measured by host |
| `timing.setup_ms` | int | Resource setup (filesystem, limits, network) |
| `timing.boot_ms` | int | VM boot time |
| `timing.execute_ms` | int | Code execution |
| `timing.total_ms` | int | End-to-end time |
| `warm_pool_hit` | bool | Whether a pre-started VM was used |
| `exposed_ports` | list | Port mappings with `.internal`, `.external`, `.host`, `.url` |

Exit codes follow Unix conventions: 0 = success, >128 = killed by signal N where N = exit_code - 128 (e.g., 137 = SIGKILL, 139 = SIGSEGV), -1 = internal error (could not retrieve status), other non-zero = program error.

```python
result = await scheduler.run(code="...", language="python")

if result.exit_code == 0:
    pass  # Success
elif result.exit_code > 128:
    signal_num = result.exit_code - 128  # e.g., 9 for SIGKILL
elif result.exit_code == -1:
    pass  # Internal error (see result.stderr)
else:
    pass  # Program exited with error
```

## FileInfo

Returned by `Session.list_files()`.

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | File or directory name |
| `is_dir` | bool | True if entry is a directory |
| `size` | int | File size in bytes (0 for directories) |

## Exceptions

| Exception | Description |
|-----------|-------------|
| `SandboxError` | Base exception for all sandbox errors |
| `TransientError` | Retryable errors — may succeed on retry |
| `PermanentError` | Non-retryable errors |
| `InputValidationError` | Caller-bug errors — bad input, session stays alive |
| `CodeValidationError` | Empty, whitespace-only, or null-byte code |
| `EnvVarValidationError` | Invalid env var names/values (control chars, size) |
| `VmTimeoutError` | VM boot timed out |
| `VmCapacityError` | VM pool at capacity |
| `VmConfigError` | Invalid VM configuration |
| `SessionClosedError` | Session already closed |
| `CommunicationError` | Guest communication failed |
| `GuestAgentError` | Guest agent returned error |
| `PackageNotAllowedError` | Package not in allowlist |
| `SnapshotError` | Snapshot operation failed |
| `SocketAuthError` | Socket peer authentication failed |
| `SandboxDependencyError` | Optional dependency missing (e.g., aioboto3) |
| `AssetError` | Asset download/verification failed |

## Session Resilience

Sessions survive user code failures and input validation errors (`InputValidationError`). Only VM-level communication errors close a session.

| Failure | Exit Code | Session | State | Next `exec()` |
|---------|-----------|---------|-------|----------------|
| Exception (ValueError, etc.) | 1 | Alive | Preserved | Works, state intact |
| `sys.exit(n)` | n | Alive | Preserved | Works, state intact |
| Syntax error | 1 | Alive | Preserved | Works, state intact |
| `os._exit(n)` | n | Alive | **Reset** | Works, fresh REPL |
| Signal (SIGKILL, OOM kill) | 128 + signal | Alive | **Reset** | Works, fresh REPL |
| Timeout | -1 | Alive | **Reset** | Works, fresh REPL |
| VM communication failure | N/A | **Closed** | Lost | `SessionClosedError` |

## Pitfalls

```python
# run() creates a fresh VM each time - state doesn't persist across calls
result1 = await scheduler.run("x = 42", language="python")
result2 = await scheduler.run("print(x)", language="python")  # NameError!
# Fix: use sessions for multi-step stateful execution
async with await scheduler.session(language="python") as session:
    await session.exec("x = 42")
    result = await session.exec("print(x)")  # Works! x persists

# Pre-started VMs (warm pool) only work without packages
config = SchedulerConfig(warm_pool_size=1)
await scheduler.run(code="...", packages=["pandas==2.2.0"])  # Bypasses warm pool, fresh start (400ms)
await scheduler.run(code="...")                        # Uses warm pool (1-2ms)
await scheduler.run(code="...", on_boot_log=print)     # Bypasses warm pool (needs cold boot for logs)

# Version specifiers are required (security + caching)
packages=["pandas==2.2.0"]  # Valid, cacheable
packages=["pandas"]         # PackageNotAllowedError! Must pin version

# Streaming callbacks must be fast (blocks async execution)
on_stdout=lambda chunk: time.sleep(1)        # Blocks!
on_stdout=lambda chunk: buffer.append(chunk)  # Fast (same applies to on_boot_log)

# Memory overhead: pre-started VMs use warm_pool_size × 3 languages × 192MB
# warm_pool_size=5 → 5 VMs/lang × 3 × 192MB = 2.88GB for warm pool alone

# Memory can exceed configured limit due to compressed swap
default_memory_mb=256  # Code can actually use ~280-320MB thanks to compression
# Don't rely on memory limits for security - use timeouts for runaway allocations

# Network without domain restrictions is risky
allow_network=True                              # Full internet access
allow_network=True, allowed_domains=["api.example.com"]  # Controlled

# Port forwarding binds to localhost only
expose_ports=[8080]  # Binds to 127.0.0.1, not 0.0.0.0
# If you need external access, use a reverse proxy on the host

# multiprocessing.Pool works, but single vCPU means no CPU-bound speedup
from multiprocessing import Pool
Pool(2).map(lambda x: x**2, [1, 2, 3])  # Works (cloudpickle handles lambda serialization)
# For CPU-bound parallelism, use multiple VMs via scheduler.run() concurrently instead

# Background processes survive across session exec() calls — state accumulates
async with await scheduler.session(language="python") as session:
    await session.exec("import subprocess; subprocess.Popen(['sleep', '300'])")
    await session.exec("import subprocess; subprocess.Popen(['sleep', '300'])")
    # Both sleep processes are still running! VM process limit (RLIMIT_NPROC=1024) prevents unbounded growth
    # All processes are cleaned up when session.close() destroys the VM
```

## Limits

| Resource | Limit |
|----------|-------|
| Max code size | 1MB |
| Max stdout | 1MB |
| Max stderr | 100KB |
| Max packages | 50 |
| Max env vars | 100 |
| Max exposed ports | 10 |
| Max file size (I/O) | 500MB |
| Max file path length | 4096 bytes (255 per component) |
| Execution timeout | 1-300s |
| VM memory | 128MB minimum (no upper bound) |
| Max concurrent VMs | Resource-aware (auto-computed from host memory + CPU) |

## Security Architecture

| Layer | Technology | Protection |
|-------|------------|------------|
| 1 | Hardware virtualization (KVM/HVF) | CPU isolation enforced by hardware |
| 2 | Custom hardened kernel | Modules disabled at compile time, io_uring compiled out, slab/memory hardening, ~360+ subsystems removed |
| 3 | Unprivileged QEMU | No root privileges, minimal exposure |
| 4 | Non-root REPL (UID 1000) | Blocks mount, ptrace, raw sockets, kernel modules |
| 5 | System call filtering (seccomp) | Blocks unauthorized OS calls |
| 6 | Resource limits (cgroups v2) | Memory, CPU, process limits |
| 7 | Process isolation (namespaces) | Separate process, network, filesystem views |
| 8 | Security policies (AppArmor/SELinux) | When available |
| 9 | Socket authentication (SO_PEERCRED/LOCAL_PEERCRED) | Verifies QEMU process identity |

**Guarantees:**

- Fresh VM per `run()`, destroyed immediately after. Sessions reuse the same VM across `exec()` calls (same isolation, persistent state)
- Network disabled by default - requires explicit `allow_network=True`
- Domain allowlisting with 3-layer outbound filtering — DNS resolution blocked for non-allowed domains, TLS SNI inspection on port 443, and DNS cross-validation to prevent SNI spoofing
- Package validation - only top 10k Python/JavaScript packages allowed by default
- Port forwarding isolation - when `expose_ports` is used without `allow_network`, guest cannot initiate any outbound connections (all outbound TCP/UDP blocked)

## Requirements

| Requirement | Supported |
|-------------|-----------|
| Python | 3.12, 3.13, 3.14 (including free-threaded) |
| Linux | x64, arm64 |
| macOS | x64, arm64 |
| QEMU | 8.0+ |
| Hardware acceleration | KVM (Linux) or HVF (macOS) recommended, ~5-8x faster |

Verify hardware acceleration is available:

```bash
ls /dev/kvm              # Linux
sysctl kern.hv_support   # macOS
```

Without hardware acceleration, QEMU uses software emulation (TCG), which is ~5-8x slower.

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
| `raw-base` | Bash | None | ~15MB | Shell scripts and custom runtimes |

All images are based on **Alpine Linux 3.23** (Linux 6.18, musl libc) and include common tools for AI agent workflows.

### Common Tools (all images)

| Tool | Purpose |
|------|---------|
| `git` | Version control, clone repositories |
| `curl` | HTTP requests, download files |
| `jq` | JSON processing |
| `bash` | Shell scripting |
| `coreutils` | Standard Unix utilities (ls, cp, mv, etc.) |
| `grep` | GNU grep (PCRE via `-P`, `--include`, etc.) |
| `findutils` | GNU find (`-printf`, `-regex`, etc.) |
| `sed` | GNU sed (in-place editing, extended regex) |
| `gawk` | GNU awk (`gensub`, `strftime`, FPAT, etc.) |
| `diffutils` | GNU diff/diff3/sdiff (`--color`, unified format) |
| `patch` | Apply unified diffs |
| `less` | Pager (backward scroll, search) |
| `make` | Build automation |
| `tree` | Directory structure visualization |
| `tar`, `gzip`, `unzip` | Archive extraction |
| `file` | File type detection |

### Python Image

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.14 | [python-build-standalone](https://github.com/astral-sh/python-build-standalone) (musl) |
| uv | 0.9+ | 10-100x faster than pip ([docs](https://docs.astral.sh/uv/)) |
| gcc, musl-dev | Alpine | For C extensions (numpy, pandas, etc.) |
| cloudpickle | 3.1.2 | Serialization for `multiprocessing` in REPL ([docs](https://github.com/cloudpipe/cloudpickle)) |

**Usage notes:**
- Use `uv pip install` instead of `pip install` (pip not included)
- Python 3.14 includes t-strings, deferred annotations, free-threading support
- `multiprocessing.Pool` works out of the box — cloudpickle handles serialization of REPL-defined functions, lambdas, and closures. Single vCPU means no CPU-bound speedup, but I/O-bound parallelism and `Pool`-based APIs work correctly

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
- Shell script execution (`language="raw"`) — runs under **GNU Bash**, full bash syntax supported
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

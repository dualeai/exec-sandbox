# exec-sandbox

[![CI](https://github.com/dualeai/exec-sandbox/actions/workflows/test.yml/badge.svg)](https://github.com/dualeai/exec-sandbox/actions/workflows/test.yml)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/dualeai/exec-sandbox?utm_source=badge)
[![Coverage](https://img.shields.io/codecov/c/github/dualeai/exec-sandbox)](https://codecov.io/gh/dualeai/exec-sandbox)
[![PyPI](https://img.shields.io/pypi/v/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![Python](https://img.shields.io/pypi/pyversions/exec-sandbox)](https://pypi.org/project/exec-sandbox/)
[![License](https://img.shields.io/pypi/l/exec-sandbox)](https://opensource.org/licenses/Apache-2.0)

exec-sandbox runs untrusted Python, JavaScript or shell code in QEMU microVMs on your own Linux or macOS hosts, for example code that an AI agent wrote. It is a Python asyncio library with a command line (`sbx`). It needs no cloud account and no API key.

```python
import asyncio

from exec_sandbox import Language, Scheduler, VmSpec


async def main() -> None:
    async with Scheduler() as scheduler:
        # Language.PYTHON, Language.JAVASCRIPT or Language.RAW
        async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
            result = await session.exec("print('Hello from a secure VM!')")
            print(result.stdout)  # Hello from a secure VM!
            print(result.exit_code)  # 0


asyncio.run(main())
```

The other Python examples use `async with` and `await` at the top level: run them in an `async def` function, as `main()` above.

## Contents

- **Start:** [Features](#features) · [Install](#install) · [Quick start](#quick-start) · [Key terms](#key-terms)
- **Use:** [Sessions](#sessions) · [Files](#files) · [Packages](#packages) · [Network](#network) · [Saved instances](#saved-instances) · [Configuration](#configuration) · [Errors and recovery](#errors-and-recovery)
- **Operate:** [Performance](#performance) · [Caches and disk use](#caches-and-disk-use) · [VM images and offline use](#vm-images-and-offline-use) · [Linux host setup](#linux-host-setup) · [Instance store operations](#instance-store-operations)
- **Reference:** [CLI reference](#cli-reference) · [API reference](#api-reference) · [Limits](#limits) · [Guest environment](#guest-environment)
- **Internals:** [How it works](#how-it-works) · [Security](#security) · [Contributing](#contributing) · [License](#license)

## Features

- **Hardware isolation.** Each session runs in its own QEMU VM, not in a container, with KVM or HVF acceleration. Up to 8 layers of isolation ([Security](#security)).
- **Fast start.** Under 1 ms from the warm pool, about 50-60 ms from an L1 memory snapshot. L2 disk snapshots keep the installed packages, and L3 shares them in S3 with all hosts ([Performance](#performance)).
- **Self-hosted.** The first use downloads the VM images from GitHub Releases. For hosts with no internet access, use `sbx prefetch`.
- **Python API and CLI.** Run code once or in several steps, stream stdout and stderr, and read and write files.
- **Saved instances.** `save()` writes the RAM, the processes and `/home/user` of a VM, encrypted, to S3 or a local folder. `reuse()` continues it later on any worker of the same host kind.
- **Network control.** No network access by default. Per spec, an allow-list of domains, checked by DNS filtering and TLS SNI inspection.
- **Memory density.** zram in the guest. On Linux, KSM page merging and memory reclaim of idle warm-pool VMs.

## Install

- **Python:** 3.12, 3.13, 3.14 (including free-threaded).
- **OS:** Linux or macOS, x64 or arm64. On Windows, use WSL2 (on native Windows, `import exec_sandbox` raises `ConfigError` `PLATFORM_UNSUPPORTED`).
- **QEMU:** 11.0+ for L1 memory snapshots and saved instances; 8.0+ for the rest. ARM64 without KVM or HVF needs 9.0.4+.
- **Hardware acceleration:** KVM (Linux) or HVF (macOS), recommended (~5-8x faster).
- **x86 CPU with KVM or HVF:** x86-64-v3 (Intel Haswell, AMD Zen 1 or newer). On a CPU without it, the `Scheduler` start raises `ConfigError` `HOST_DEPENDENCY` with the missing features. TCG runs on any x86 CPU.

### Install the package

```bash
uv add exec-sandbox              # Core library
uv add "exec-sandbox[cli]"       # + the sbx command line and the local instance store
uv add "exec-sandbox[s3]"        # + the L3 snapshot cache and the s3 instance store (and the local store)
```

With pip, use `pip install` with the same names. For `sbx` with the `s3` store, install `exec-sandbox[cli,s3]`.

### Install QEMU

```bash
brew install qemu                             # macOS
apt install qemu-system qemu-utils            # Ubuntu/Debian
test -r /dev/kvm -a -w /dev/kvm && echo ok    # Linux: ok means your user can use KVM
sysctl kern.hv_support                        # macOS: kern.hv_support: 1 means HVF is available
```

exec-sandbox uses `qemu-system-x86_64` or `qemu-system-aarch64` from `/usr/bin/`, else from `PATH` (`EXEC_SANDBOX_QEMU_BIN_X86` and `EXEC_SANDBOX_QEMU_BIN_ARM` set another file), and `qemu-img` and `qemu-storage-daemon` from `PATH`. A missing QEMU binary, or one whose version or machine types this user cannot read, raises `ConfigError` (`HOST_DEPENDENCY`) when the `Scheduler` starts; a missing `qemu-img` raises it at its first use. Check the version with `/usr/bin/qemu-system-x86_64 --version` (or `qemu-system-aarch64`): Ubuntu 24.04 ships 8.2.2. Saved instances and L1 snapshots need QEMU 11.0 or later: where the distribution ships an older one, build QEMU from its source release ([qemu.org/download](https://www.qemu.org/download/#source)) and set `EXEC_SANDBOX_QEMU_BIN_X86` or `EXEC_SANDBOX_QEMU_BIN_ARM` to the binary.

On Linux, give your user access to KVM with `sudo usermod -aG kvm $USER`, then log in again. Without this access, exec-sandbox logs a warning and uses TCG. To run QEMU as its own user, with cgroup limits and namespaces, see [Linux host setup](#linux-host-setup).

## Quick start

The first run downloads the VM images ([How images download](#how-images-download)). For Python, see the example at the top and [Sessions](#sessions). `sbx` comes with the `cli` extra; all its options are in the [CLI reference](#cli-reference).

```bash
sbx run 'print("Hello from sandbox")'                                  # Python
sbx run -l javascript 'console.log("Hello from sandbox")'              # JavaScript
sbx run script.py                                                      # A file: the extension gives the language
sbx run -p pandas==3.0.1 'import pandas; print(pandas.__version__)'    # With packages
sbx run --network --allow-domain api.example.com fetch_data.py         # Network, with an allow-list
sbx run --instance my-vm -c 'x = 41'                                   # Keep the VM state (Saved instances)
sbx run --instance my-vm -c 'print(x + 1)'                             # 42
```

## Key terms

| Term | Meaning |
|------|---------|
| Spec | A `VmSpec`: what a VM is made of ([VmSpec](#vmspec)) |
| REPL | The interpreter in the VM that runs your code |
| Session | One VM that runs your code: `new()` or `reuse()` gives it, `close()` ends the VM ([Sessions](#sessions)) |
| Saved instance | The state of a session that `save()` stored (RAM, processes, files in `/home/user`): `reuse()` continues it on any worker of its host kind ([Saved instances](#saved-instances)) |
| Warm pool | VMs started before the request ([Warm pool](#warm-pool)) |
| L1, L2, L3 | The snapshot cache: memory snapshots, disk snapshots with the packages installed, and a copy of L2 in S3 ([Snapshot cache](#snapshot-cache)). It makes new VMs start faster; it holds no saved instance |
| Instance store | Where saved instances live: your S3 bucket (`s3`) or a folder on this host (`local`) ([Choose a store](#choose-a-store)) |
| Generation | The save number of an instance: 1, then +1 for each save |
| Head | The object in the store that names the current state of an instance |
| VM images | The kernel, the initramfs and the base disk image of each language ([VM images and offline use](#vm-images-and-offline-use)) |
| Assets | The VM images and the gvproxy-wrapper binary: what `sbx prefetch` downloads |
| Guest images | The disk images of one VM: the base image and, with packages, the L2 package image |
| Worker | A process that runs a `Scheduler`, on its host (a machine or a container) |
| Host kind | What a saved instance must match to load ([Rules for save and reuse](#rules-for-save-and-reuse)) |
| Admission | The wait until the host has the memory, CPU and disk for a VM ([Limits](#limits)) |
| KVM, HVF, TCG | Hardware acceleration on Linux (KVM) and macOS (HVF). TCG is QEMU's software emulation |

## Sessions

`scheduler.new(spec)` starts a new VM and returns its session. The session keeps variables, imports, files and background processes between `exec()` calls ([Limits](#limits)), until `close()` destroys the VM. Some errors close a session ([What closes a session](#what-closes-a-session)).

```python
async with Scheduler() as scheduler:
    async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
        await session.exec("import math")
        await session.exec("x = math.pi * 2")
        result = await session.exec("print(f'{x:.4f}')")
        print(result.stdout)  # 6.2832
        print(session.exec_count)  # 3
```

JavaScript sessions (also TypeScript) keep variables and functions. Shell sessions (`raw`, Bash) keep environment variables, the working folder and functions.

A session closes after `session_idle_timeout_seconds` (default 300) with no call. For one session, pass `idle_timeout_seconds` (10-3600) to `new()` or `reuse()`; a value out of range raises `InvalidInputError` (`ARGUMENT_INVALID`).

To get the output while the code runs, pass `on_stdout` and `on_stderr` to `exec()`: `await session.exec(code, on_stdout=lambda chunk: print(chunk, end=""))`. To get the boot output of the kernel, tiny-init and the guest agent, pass `on_boot_log` to `new()`; the VM then always cold boots (no warm pool, no L1). Keep each callback short: a slow callback blocks the event loop.

## Files

Paths are relative to `/home/user`, the folder where your code runs.

```python
await session.write_file("input.csv", b"name,score\nAlice,95")  # From bytes, a local Path or an IO[bytes] stream
await session.write_file("run.sh", Path("./run.sh"), make_executable=True)  # Mode 0755 (default 0644)
await session.exec("open('output.csv', 'w').write(open('input.csv').read().upper())")
await session.read_file("output.csv", destination=Path("./output.csv"))  # To a local Path or an IO[bytes] stream
for f in await session.list_files(""):  # "" is /home/user
    print(f.name, f.is_dir, f.size)
```

```bash
sbx run --upload ./local.csv:input.csv --download output.csv:./result.csv -c "open('output.csv','w').write(open('input.csv').read().upper())"
```

## Packages

Set `packages` in the spec. The first VM of a spec installs the packages and caches the result; later VMs of that spec start from the cache ([Snapshot cache](#snapshot-cache)). Count and size limits: [Limits](#limits). Runtimes and tools: [Guest environment](#guest-environment).

- Give each package a version: `VmSpec(language=Language.PYTHON, packages=("pandas==3.0.1",))`, `VmSpec(language=Language.JAVASCRIPT, packages=("lodash@4.17.21",))`. Other operators (`>=`, `~=`) pass too; `==` gives a stable cache entry.
- A package without a version, a bad name, or packages for `raw` raise `InvalidInputError` (`ARGUMENT_INVALID`). Scoped npm names (`@scope/name`) are refused.
- A package outside the allow-list (the top 10,000 of PyPI or npm) raises `InvalidInputError` (`PACKAGE_NOT_ALLOWED`). To skip this check, set `enable_package_validation=False` (`sbx run --no-validation`).
- Python packages install into Python 3.14 on musl with no build step: use packages with pure-Python or musllinux wheels.
- The install reaches PyPI or npm, whatever the network settings of the spec.

## Network

### Outbound access

`allow_network=True` turns on outbound access. `allowed_domains` then sets what the guest can reach:

- `None` (default): the package registries of the language (python: `pypi.org`, `files.pythonhosted.org`; javascript: `registry.npmjs.org`). `raw` reaches every host.
- `()` (empty): every host.
- A list of domains, such as `("httpbin.org",)`: these domains and their subdomains (`example.com` also allows `api.example.com`).

### Port forwarding

```python
from exec_sandbox import Language, PortMapping

# Guest port 8080 on host port 3000
spec = VmSpec(language=Language.PYTHON, expose_ports=(PortMapping(internal=8080, external=3000),))
async with await scheduler.new(spec) as session:
    print(session.exposed_ports[0].url)  # http://127.0.0.1:3000
```

- A fixed host port must be 1024-65535. An int in place of a `PortMapping` (`expose_ports=(8080,)`) is the guest port: the OS picks the host port. `protocol` is `"tcp"` (default) or `"udp"`.
- Ports bind to 127.0.0.1. For access from other machines, put a reverse proxy on the host.
- Port forwarding works without `allow_network`; all outbound TCP and UDP then stays blocked.
- After `reuse()`, the ports open again: a fixed host port must be free on this host, and a dynamic one can change (read `exposed_ports` again).

### Optional private access for a reviewer device

[qURL](https://qurl.link/?utm_source=exec-sandbox-github&utm_medium=referral&utm_campaign=first-share-pilot)
is a hosted option for a person or agent with the qURL CLI to fetch an HTTP app
through its host-loopback port. Identity comes before connection: only approved
devices can obtain access links, and the app needs no public inbound port.
This uses LayerV's service, not a self-hosted exec-sandbox ingress provider.

Install the [qURL CLI](https://github.com/layervai/qurl-integrations/tree/main/apps/cli)
on both machines. On the reviewer's machine, get its public device key:

```sh
qurl whoami -o json
```

Send only `device_public_key_b64` to the publisher through a trusted channel.
Keep the sandbox session and guest HTTP server running. On the host, publish
its current `session.exposed_ports[0].url` (3000 is the example above):

```sh
qurl publish http://127.0.0.1:3000 --id sandbox-review-1 --name "Sandbox review" --allow-device-key '<reviewer-public-key>' -o json
```

Use a fresh local ID. Check that `found_existing` is false and `private`
is true before continuing; a retry can reuse an existing resource and policy.
Do not combine this device-key method with `--access` or the separate browser
email/passkey approval workflow.

Send the returned CRID to the reviewer. On their machine:

```sh
qurl get '<CRID>' --file -
```

This prints the HTTP response body. Check that it is the expected app content.
From an interactive terminal, `qurl get '<CRID>'` opens the access link in
the reviewer's browser instead.

To stop issuing new links to that device, run
`qurl grants '<CRID>' --remove '<reviewer-public-key>'`. Already-issued links
keep their own expiry. Delete the share with `qurl delete '<CRID>'` **before**
closing or reusing the sandbox session: its host port can later belong to a
different app. This is a manual workflow; it does not automate VM lifecycle
cleanup or implement `ExposedPort.public_url`.

## Saved instances

`save()` writes the state of a VM (RAM, processes, files in `/home/user`), compressed and encrypted, to the instance store. `reuse()` continues it later, on any worker of the same host kind. Every worker needs QEMU 11.0 or later (else the `Scheduler` start fails with `ConfigError` `MIGRATION_UNSUPPORTED`) and a keyring: set `EXEC_SANDBOX_INSTANCE_KEYRING`, or pass `instance_keyring=` in code ([Keys](#keys)).

```python
import os

from exec_sandbox import Language, Scheduler, SchedulerConfig, VmSpec

KEYRING = os.environ["MY_KEYRING"]  # from your secret store: see Keys
config = SchedulerConfig(instance_store="local", instance_keyring=KEYRING)  # "s3" for several workers
async with Scheduler(config) as scheduler:
    async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
        await session.exec("x = 41")
        ref = await session.save()  # InstanceRef(instance_id=..., generation=1)
    async with await scheduler.reuse(ref.instance_id) as session:  # Later, with the same keyring
        result = await session.exec("x += 1; print(x)")
        print(result.stdout)  # 42
        await session.save()  # generation 2
```

`save()` keeps the VM running: the session goes on after the save.

### Keys

Make a key with `python3 -c "import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"`. The keyring is a comma-separated list of keys, newest first: the first key encrypts new saves, the others only open older instances. Keep the keyring in your secret store, and give every worker the same keyring. Keep every key that saved an instance: without it, `reuse()` raises `ConfigError` (`INSTANCE_KEY_MISSING`). To change keys: [Key rotation](#key-rotation).

### Instance ids

To choose the id, pass it to `new()`: `scheduler.new(spec, instance_id="user-42")`. An id has 1 to 64 lowercase letters, digits, `.`, `_` or `-`, and starts with a letter or digit. If the id exists, the first `save()` fails with `INSTANCE_CONFLICT`. Without an id, the first save makes a random one (`ref.instance_id`).

### Save under a new id

`session.save(instance_id="run-17")` saves the VM as generation 1 of a new instance, then the session saves to that instance (`session.instance`). The instance the session had stays as its last save left it. An id that exists raises `INSTANCE_CONFLICT`, and nothing is overwritten. The session's own id gives a plain save.

To start many sessions from one save, call `reuse()` of the same instance in each, and give each session its own id at its first save:

```python
children = [await scheduler.reuse(checkpoint.instance_id) for _ in range(3)]
for n, child in enumerate(children):
    await child.exec(f"attempt({n})")
    await child.save(instance_id=f"attempt-{n}")  # the checkpoint stays as it was
```

### Random values in copies

Each VM that starts from the same save gets new values from its first `exec()`: the kernel random source (`os.urandom`, `/dev/urandom`), Python `ssl` and the module functions of `random`, the random functions of the JavaScript `crypto` and `node:crypto`, `Math.random` (also in the `node:vm` contexts that the code makes), the keys of the `Sync` functions of `node:crypto` (such as `generateKeyPairSync`), and bash `$RANDOM`. Python `random` gets a new seed at each `reuse()`: to repeat a seeded run, call `random.seed()` after the load. These keep the state of the save in each copy: values made before the save, programs that run since before the save, `random.Random()` objects and NumPy generators. In Bun, these can give the same values in each copy: the keys, primes and signatures of the functions that take a callback or return a promise (such as `generateKeyPair`, `generatePrime` and `crypto.subtle`), TLS and `fetch`, and Workers. To get other values in each copy, make keys with the `Sync` functions, and seed the other generators in the copy.

### Choose a store

The `local` store (`instance_store="local"`) is a folder on this host, for one host. For the `s3` store, set up the bucket ([Set up the S3 bucket](#set-up-the-s3-bucket)), then give every worker the same keyring and the same config: `SchedulerConfig(instance_store="s3", s3_bucket="my-bucket")`. At start, the `Scheduler` checks the bucket and logs "Instance store ready" with its key ids.

### Use instances from the command line

`sbx run --instance ID` continues the instance `ID`, or makes it when it does not exist:

```bash
sbx run --instance my-vm -c 'x = 41'            # Created instance my-vm, generation 1
sbx run --instance my-vm -c 'x += 1; print(x)'  # 42, then: Saved instance my-vm, generation 2
sbx instance list                               # One id on each line (--json for an array)
sbx instance list --long                        # One JSON object for each instance: as describe, or its error
sbx instance describe my-vm                     # JSON: generation, recent commits, key id, access, writer release; spec, host (with the device layout), guest and egress domains when access is full
sbx instance keys                               # The id of each key of the keyring, newest first
sbx instance check my-vm                        # "my-vm ok", or why this worker cannot continue it
sbx worker describe                             # JSON: this worker's facts (WorkerFacts), for a broker
sbx instance rm my-vm other-vm                  # In order, stops at the first error; exit code 0 when they are gone
```

- **Store.** `sbx` reads the environment variables of [SchedulerConfig](#schedulerconfig). By default, it uses the local store in `instance_store_dir` (`EXEC_SANDBOX_INSTANCE_STORE_DIR`). For S3, set `EXEC_SANDBOX_INSTANCE_STORE=s3`, `EXEC_SANDBOX_S3_BUCKET` and `EXEC_SANDBOX_INSTANCE_KEYRING`, and the other `EXEC_SANDBOX_S3_*` variables as needed. Only `--instance` and `sbx instance` open the store.
- **Key.** `sbx` reads `EXEC_SANDBOX_INSTANCE_KEYRING`; set it on a host with no OS keyring, such as a server or a container. Else, for the local store, `sbx` makes one key at the first use and keeps it in the OS keyring (macOS Keychain; Secret Service or KWallet on Linux; no other `keyring` backend), entry `exec-sandbox` / `instance-keyring`. Back it up with `keyring get exec-sandbox instance-keyring`, restore it with `keyring set exec-sandbox instance-keyring` (the `keyring` command of the `cli` extra, in the same Python environment). To open these instances from code, pass `instance_keyring=keyring.get_password("exec-sandbox", "instance-keyring")`.
- **Save.** A run saves after the uploads, the code and the downloads, also when the code exits with an error or a download fails. It saves nothing after a timeout, a VM that stopped, or a failed upload or run; stderr tells why. `-q` hides the "Created/Saved" line.
- **VM settings.** `-l`, `-p`, `-m`, `--network`, `--allow-domain` and `--expose` are fixed when the instance is made. Leave them out or give the same value; another value exits with code 2. With `--allow-domain`, also give `--network`. `--instance` takes one source (`-c`, a file or `-`).
- **Exit codes.** A bad id exits with code 2. If two runs save at the same time, the second save fails with `INSTANCE_CONFLICT` and exit code 125; its output stays, its state does not. Others: [Exit codes](#exit-codes).

### List, describe and delete instances

```python
info = await scheduler.describe_instance(
    instance_id
)  # Generation, recent commits, key id; spec and host when this keyring opens it
deleted = await scheduler.delete_instance(instance_id)  # True if it existed
async for instance_id in scheduler.list_instances():
    ...
```

To do this without a `Scheduler` (no VM, no guest image, no QEMU: a cleanup job on any host), open the store with `instance_admin`:

```python
from exec_sandbox import SchedulerConfig, instance_admin

async with instance_admin(SchedulerConfig(instance_store="s3", s3_bucket="my-bucket")) as admin:
    async for instance_id in admin.list_ids():
        info = await admin.describe(instance_id)
```

### Rules for save and reuse

- **One writer.** `save()` fails with `INSTANCE_CONFLICT` if another save of the instance was stored since this session loaded or last saved it. The session stays open: save it under a new id to keep its work, or close it. If two requests can reach one instance at the same time, use your own lock.
- **Network.** A reused instance reaches the domains its last save allowed (`describe_instance(id).egress_domains`; empty: all), also after a release that changes the default list of `allowed_domains=None`.
- **`close()` never saves.** After an `exec()` or `write_file()` with no later save, `close()` warns with `InstanceNotSavedWarning` (not after another save of the instance won: no plain save of that session can store its work). The idle timeout closes the session with no save (it logs a warning in the cases where `close()` warns): save before a long pause.
- **Host kind.** An instance loads only on a worker with the same CPU architecture, accelerator (`kvm`, `hvf` or `tcg`), guest CPU model, host CPU (ARM with KVM or HVF) and guest clock rate (any rate on x86 KVM with TSC scaling), and a QEMU that has the saved machine type (for example `virt-11.1`) and accepts the saved state. The release of the worker must run the saved devices (`describe_instance(id).host.device_layout`). Other workers raise `RouteError`; its `required` field gives the values (empty for the devices: load it on a worker of another release).
- **After `reuse()`.** The network connections that were open at the save are closed.

### Errors of saved instances

| `action` (class) | Codes | Do this |
|------------------|-------|---------|
| `RETRY` (`TransientError`) | `INSTANCE_STORE_UNAVAILABLE`, `LOCAL_DISK` | Call again later. A failed save stored nothing. |
| `ROUTE` (`RouteError`) | `INSTANCE_HOST_MISMATCH` | Use a worker that satisfies `required` ([Upgrades](#upgrades)). Keep the instance. |
| `RECREATE` (`InstanceGoneError`) | `INSTANCE_NOT_FOUND`, `INSTANCE_EXPIRED`, `INSTANCE_INTEGRITY` | Start a new instance. To use the same id again, call `delete_instance(id)` first. A save that raises `INSTANCE_NOT_FOUND` keeps its session open: save it under a new id to keep its work. |
| `VERIFY` (`OutcomeUnknownError`) | `INSTANCE_SAVE_OUTCOME_UNKNOWN` | The session stays open. The save was stored if `describe_instance(id).recent_commits` has `error.context["commit_id"]`: check before you do the work again. A retry of the same save (the same `instance_id`) never stores a second copy. |
| `VERIFY` (`OutcomeUnknownError`) | `INSTANCE_CONFLICT` | This save was not stored, and the session stays open. If an earlier save of the session had an unknown outcome, look for its `commit_id` in `describe_instance(id).recent_commits` first: it may be the save that is stored. Else another save holds the instance: save this session under a new id to keep its work, or `reuse()` the instance (it has the other save) and do the work again. For a new id, or the first save of `new(instance_id=...)`, the id is taken: use another id. |
| `FIX_CONFIG` (`ConfigError`) | `INSTANCE_CONFIG`, `INSTANCE_KEY_MISSING`, `MIGRATION_UNSUPPORTED`, `IMAGE_DAMAGED` | `INSTANCE_CONFIG`: fix the bucket, store folder, permissions or keyring, or use a store with conditional writes. `INSTANCE_KEY_MISSING`: add the named key; keep the instance. `MIGRATION_UNSUPPORTED`: upgrade QEMU to 11.0+. `IMAGE_DAMAGED`: download or rebuild the images. |
| `FIX_INPUT` (`InvalidInputError`) | `INSTANCE_INVALID_ID` | Fix the id ([Instance ids](#instance-ids)). |

`save()` and `reuse()` can also raise the other codes of `ErrorCode`: [Error codes](#error-codes).
## Configuration

### VmSpec

`VmSpec` says what a VM is made of. It is frozen and hashable: keep one spec per workload.

| Field | Default | Description |
|-------|---------|-------------|
| `language` | required | `"python"`, `"javascript"` or `"raw"` (shell) |
| `memory_mb` | 192 | VM memory, 128 MB minimum. See [Memory and file systems](#memory-and-file-systems) |
| `cpu_cores` | 1 | Guest vCPUs |
| `disk_mb` | 1024 | Home disk size (`/home/user`, ext4), 128 MB minimum. Only written blocks use host disk |
| `packages` | `()` | Packages, each with a version (`"pandas==3.0.1"`, `"lodash@4.17.21"`). See [Packages](#packages) |
| `allow_network` | False | Outbound network access |
| `allowed_domains` | None | With `allow_network`: the domains the guest can reach. See [Outbound access](#outbound-access) |
| `expose_ports` | `()` | Guest ports to expose on the host (`PortMapping` or an int), at most 10. See [Port forwarding](#port-forwarding) |

### SchedulerConfig

Settings for all VMs of a `Scheduler`: `Scheduler(SchedulerConfig(warm_pool_size=1))`. A field that you do not set in code reads its environment variable. An invalid value fails the `Scheduler` start with `ConfigError` (`SETTING_INVALID`).

| Field | Default | Env var | Description |
|-------|---------|---------|-------------|
| **Sessions and warm pool** | | | |
| `warm_pool_size` | 0 | — | Warm-pool VMs per language, for `VmSpec(language=...)` with no other field. 0: no warm pool |
| `default_timeout_seconds` | 30 | — | Execution timeout (1-300 s) |
| `session_idle_timeout_seconds` | 300 | — | Close a session after this long without a call (10-3600 s) |
| **Admission** | | | |
| `memory_overcommit_ratio` | 5.0 | `EXEC_SANDBOX_MEMORY_OVERCOMMIT_RATIO` | 1.0-10.0. Budget = host memory × (1 - reserve) × ratio |
| `cpu_overcommit_ratio` | 2.0 | `EXEC_SANDBOX_CPU_OVERCOMMIT_RATIO` | 1.0-20.0. Budget = (host CPUs - reserve) × ratio |
| `disk_overcommit_ratio` | 4.0 | `EXEC_SANDBOX_DISK_OVERCOMMIT_RATIO` | 1.0-64.0. Each VM reserves the unwritten part of its `disk_mb` / ratio |
| `host_memory_reserve_ratio` | 0.08 | `EXEC_SANDBOX_HOST_MEMORY_RESERVE_RATIO` | Fraction of host memory kept for the OS (0.01-0.5) |
| `host_cpu_reserve_cores` | 0.5 | `EXEC_SANDBOX_HOST_CPU_RESERVE_CORES` | CPU cores kept for host processes (0.0-16.0) |
| **Packages** | | | |
| `enable_package_validation` | True | — | Check packages against the allow-list (top 10,000 of PyPI and of npm) |
| **Images and caches** | | | |
| `images_dir` | None | `EXEC_SANDBOX_IMAGES_DIR` | Folder of the VM images. None: found automatically ([How images download](#how-images-download)) |
| `auto_download_assets` | True | — | Download missing VM images from GitHub Releases |
| `disk_snapshot_cache_dir` | `<cache dir>/disk-snapshots/` | `EXEC_SANDBOX_DISK_SNAPSHOT_CACHE_DIR` | L2 cache folder ([Where files go](#where-files-go)) |
| `memory_snapshot_cache_dir` | `<cache dir>/memory-snapshots/` | `EXEC_SANDBOX_MEMORY_SNAPSHOT_CACHE_DIR` | L1 cache folder |
| **S3** | | | |
| `s3_bucket` | None | `EXEC_SANDBOX_S3_BUCKET` | Bucket of the L3 snapshot cache and of the `s3` instance store |
| `s3_region` | us-east-1 | `EXEC_SANDBOX_S3_REGION` | AWS region |
| `s3_endpoint_url` | None | `EXEC_SANDBOX_S3_ENDPOINT_URL` | S3-compatible endpoint (R2, MinIO, Ceph RGW); None for AWS |
| `s3_snapshot_prefix` | snapshots/ | `EXEC_SANDBOX_S3_SNAPSHOT_PREFIX` | Key prefix of the L3 snapshot cache |
| `s3_instance_prefix` | instances/ | `EXEC_SANDBOX_S3_INSTANCE_PREFIX` | Key prefix of saved instances ([Set up the S3 bucket](#set-up-the-s3-bucket)) |
| `max_concurrent_s3_uploads` | 128 | `EXEC_SANDBOX_MAX_CONCURRENT_S3_UPLOADS` | Max concurrent background S3 uploads (1 or more) |
| **Saved instances** | | | |
| `instance_store` | None | `EXEC_SANDBOX_INSTANCE_STORE` | `s3` (`s3_bucket`, `s3` extra) or `local` (`instance_store_dir`, `cli` or `s3` extra). `None` set in code turns saved instances off |
| `instance_store_dir` | OS data folder | `EXEC_SANDBOX_INSTANCE_STORE_DIR` | Root of the `local` store. macOS: `~/Library/Application Support/exec-sandbox/instances/`. Linux: `~/.local/share/exec-sandbox/instances/` (under `$XDG_DATA_HOME` when it is an absolute path) |
| `instance_keyring` | None | `EXEC_SANDBOX_INSTANCE_KEYRING` | Required with `instance_store`: comma-separated base64 32-byte keys, newest first. Never logged or passed to child processes |

Folder rules: `instance_store_dir` must be a folder of this user, mode 0700, with no ACL entry (exec-sandbox makes it when it is missing). Only root and this user can write the parent folders of `instance_store_dir`, the cache folders and `EXEC_SANDBOX_HOME_DISK_DIR`. Sticky folders such as `/tmp` pass, and on Linux, so does a folder that only this user's private group can write. For a root volume that others can write (for example a Kubernetes `emptyDir`), set the sticky bit (`chmod +t`) as root, for example in an init container.

### Environment variables

Variables that are not `SchedulerConfig` fields:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXEC_SANDBOX_CACHE_DIR` | OS cache folder | Root of the cache folders ([Where files go](#where-files-go)) |
| `EXEC_SANDBOX_HOME_DISK_DIR` | `<cache dir>/home-disks` | Home disks of running VMs, and work files of `save()` and `reuse()`. A folder of this user that group and others cannot write, with no ACL entry, on a disk (not tmpfs) |
| `EXEC_SANDBOX_INSTANCE_ASSET_DIR` | `<cache dir>/instance-assets` | Guest images downloaded from the instance store |
| `EXEC_SANDBOX_DISK_FREE_FLOOR_MB` | 1024 | Free space that admission keeps on the disk of the home disk folder |
| `EXEC_SANDBOX_AVAILABLE_MEMORY_FLOOR_MB` | 0 | Available host memory that admission keeps free |
| `EXEC_SANDBOX_HOST_MEMORY_MB`, `EXEC_SANDBOX_HOST_CPU_COUNT` | detected | Host size for the admission budgets |
| `EXEC_SANDBOX_QEMU_BIN_X86`, `EXEC_SANDBOX_QEMU_BIN_ARM` | `/usr/bin/qemu-system-x86_64`, `/usr/bin/qemu-system-aarch64` | QEMU binary ([Install QEMU](#install-qemu)) |
| `EXEC_SANDBOX_FORCE_EMULATION` | false | Use TCG even when KVM or HVF is available |
| `EXEC_SANDBOX_SNAPSHOT_CACHE_TTL_DAYS` | 14 | Value of the `ttl_days` tag on L3 uploads |
| `EXEC_SANDBOX_OFFLINE` | unset | `1`: never download assets (a missing asset fails) |
| `EXEC_SANDBOX_ASSET_VERSION` | unset | Use the assets of this release version |
| `EXEC_SANDBOX_LOG_LEVEL` | unset | Level of the `exec_sandbox` logger (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `GITHUB_TOKEN` | unset | Sent to the GitHub API for asset downloads (higher rate limit) |

The library logs to the `exec_sandbox` logger and adds no handler. `configure_logging(level="INFO")` adds a stderr handler (the CLI does this).

## Errors and recovery

### Error classes

Every run-time error is a `SandboxError` with an `action` (what to do next, from its class), a `code` (an `ErrorCode`), a `message`, a `hint`, an `instance_id` and a `context`. After a QEMU crash or a boot timeout, `context["console_log"]` holds the VM console. A bad value in `VmSpec`, `SchedulerConfig` or `PortMapping` raises pydantic's `ValidationError`.

| Name | Action | When |
|------|--------|------|
| `SandboxError` | — | Base class (never raised itself) |
| `TransientError` | `RETRY` | The call did nothing you rely on (exec: the code was not sent). Repeat it, on this worker or another, a bounded number of times (for example 3, with backoff). If the session closed, redo its unsaved work on a new session first |
| `OutcomeUnknownError` | `VERIFY` | The code may have run, or a save's outcome is unknown, or another save was stored first. Check the state first |
| `SessionClosedError` | `RESTART` | The session is closed. Open a new one with `new()` or `reuse()` |
| `InvalidInputError` | `FIX_INPUT` | The call is wrong, or the scheduler cannot take it now. The session stays open |
| `ConfigError` | `FIX_CONFIG` | This worker's setup is wrong: settings, keyring, store, binaries, images, Python extras, permissions or OS |
| `RouteError` | `ROUTE` | Another worker can serve the call: `required` gives the facts the call checked, `found` this worker's values. A worker satisfies `required` when each key holds by its rule (`RouteKey.rule`: `equal`, `at_least`, `contains` or `equal_or_any`); it passes the checks that raised the error. An empty `required` (another release wrote the instance) names no worker: use a worker of the release in `describe_instance(id).writer_release`, or in the error's `context["writer_release"]` when `describe_instance` raises it too and the data names its release |
| `InstanceGoneError` | `RECREATE` | The saved instance is gone, expired or damaged. Start a new one |
| `InternalError` | `REPORT` | A bug or a broken host. Report it with the log |
| `ErrorAction` | | The 8 actions above |
| `ErrorCode` | | See [Error codes](#error-codes) |
| `InstanceNotSavedWarning` | | Warning: `close()` of a saved-instance session with `exec()` or `write_file()` calls after its last `save()` |

`RouteKey` names the keys of `RouteError.required` and of `WorkerFacts.values` (`describe_worker()`, `sbx worker describe`); `RouteRule` names their rules. In `WorkerFacts.values`, a `contains` key holds the set of values this worker has (a sorted list in JSON), and `ROUTE_ANY` (`"any"`) is the value of an `equal_or_any` key that takes every value.

| Key | Rule | Value |
|-----|------|-------|
| `arch` | `equal` | `x86_64` or `aarch64` |
| `accelerator` | `equal` | `kvm`, `hvf` or `tcg` |
| `cpu_model` | `equal` | The guest CPU model (QEMU `-cpu` value) |
| `cpu_id` | `equal` | The host CPU an ARM guest sees with KVM or HVF; absent on x86 and ARM TCG |
| `counter_hz` | `equal_or_any` | Rate in Hz of the guest clock's counter, set at its first boot; absent when that host had no exact rate. A worker's `any`: x86 KVM with TSC scaling, which takes every rate |
| `qemu_version` | `at_least` | `MAJOR.MINOR.PATCH`, compared part by part |
| `machine_type` | `contains` | A QEMU machine type (`virt-11.1`, `microvm`) |
| `base_image` | `contains` | SHA-256 digest (64 hex characters) |
| `packages_image` | `contains` | SHA-256 digest (64 hex characters) |
| `memory_mb` | `at_least` | Whole MB; a worker's `null`: no limit |
| `cpu` | `at_least` | Cores with two decimals; a worker's `null`: no limit |

Catch the class of each action that you handle:

```python
from exec_sandbox import ErrorCode, InvalidInputError, OutcomeUnknownError, SandboxError, TransientError

try:
    async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:
        result = await session.exec("print('hello')", timeout_seconds=5)  # A timeout does not raise: exit_code -1
except InvalidInputError as e:  # FIX_INPUT: change the call
    if e.code is ErrorCode.PACKAGE_NOT_ALLOWED:
        ...
except TransientError:  # RETRY: the same call can succeed later
    ...
except OutcomeUnknownError:  # VERIFY: the code may have run
    ...
except SandboxError as e:  # The other actions
    print(f"{e.code} ({e.action}): {e}")
```

### Exit codes and timeouts

`result.exit_code` is 0 for success and -1 for a timeout. A value above 128 usually means that signal N (exit code - 128) killed the code (137 = SIGKILL, 139 = SIGSEGV). Other values mean that the program failed. It is a process status (0-255): `sys.exit(-1)` gives 255. Infrastructure failures raise an error and return no result. For the exit codes of `sbx`, see [Exit codes](#exit-codes).

### What closes a session

- **Stays open, state kept:** an exception, a syntax error, `sys.exit(n)` with n ≠ 137, and `InvalidInputError` (for example `OUTPUT_LIMIT`, unless its message ends with "the REPL then exited with code n": see the next line).
- **Stays open, the next `exec()` runs in a new REPL:** a timeout (-1), a signal other than SIGKILL (128 + signal), `os._exit(n)` with n ≠ 137 (a REPL exit or signal after output past the limit: `OUTPUT_LIMIT` whose message ends with "the REPL then exited with code n"), and a REPL exit between two calls with a code other than 137.
- **Closes, state lost:** exit code 137 (for example SIGKILL), also between two calls (the next `exec()` raises `GUEST_UNAVAILABLE`); a VM communication failure; QEMU stops; a full host disk under the home disk. Every later call raises `SessionClosedError`.
- Other errors: in `exec()`, they close the session. File calls close it only on cancellation, `OutcomeUnknownError`, a dead VM or a full host disk. For `save()`, see its docstring.

### Error codes

| Code | Class | Meaning |
|------|-------|---------|
| `VM_BOOT_TIMEOUT` | `TransientError` | The guest agent did not answer in time at boot. |
| `QEMU_CRASHED` | `TransientError` | QEMU exited or did not come up. |
| `OVERLAY_FAILED` | `TransientError` | qemu-img or qemu-storage-daemon could not make a disk image. |
| `GVPROXY_FAILED` | `TransientError` | gvproxy did not start or become ready. |
| `CAPACITY` | `TransientError` | No resources became free within the admission timeout. |
| `GUEST_UNAVAILABLE` | `TransientError` | The guest could not take the call or start the code. The code did not run. |
| `GUEST_SILENT_AFTER_RESTORE` | `TransientError` | The guest did not answer after a memory snapshot restore, or could not reseed its kernel random source. |
| `LOCAL_DISK` | `TransientError` | A local file is missing (a path that you gave: `ARGUMENT_INVALID`), or the host cannot read or write it now (disk full, quota, I/O error, file handle or memory limit), or the host disk has no room for a home disk. |
| `PACKAGE_DOWNLOAD` | `TransientError` | The package manager could not download (network error or timeout). |
| `MIGRATION_FAILED` | `TransientError` | QEMU did not save or load the VM state. |
| `ASSET_DOWNLOAD` | `TransientError` | A VM image download failed after its retries. |
| `INSTANCE_STORE_UNAVAILABLE` | `TransientError` | The instance store failed, answered badly, or kept changing. |
| `GUEST_OUTCOME_UNKNOWN` | `OutcomeUnknownError` | The transport was lost after the call reached the guest. The code may have run. |
| `INSTANCE_CONFLICT` | `OutcomeUnknownError` | Another save was stored first, or the id of a first save is taken. This save was not stored; the session stays open. |
| `INSTANCE_SAVE_OUTCOME_UNKNOWN` | `OutcomeUnknownError` | The store did not confirm the save, and a read-back could not tell. |
| `SESSION_CLOSED` | `SessionClosedError` | The session is closed. |
| `CODE_INVALID` | `InvalidInputError` | The code is empty, only spaces, holds null bytes, or is too large. |
| `ENV_VAR_INVALID` | `InvalidInputError` | An env var name or value breaks a rule. |
| `OUTPUT_LIMIT` | `InvalidInputError` | stdout or stderr went above the guest limit. |
| `PATH_INVALID` | `InvalidInputError` | A guest path breaks a rule (empty, too long, outside the sandbox, a folder). |
| `GUEST_IO` | `InvalidInputError` | The guest could not read, list or write the path (not found, no access, disk full). |
| `ARGUMENT_INVALID` | `InvalidInputError` | An argument breaks a documented limit: timeout range, content size, stream, local path, package without a version, packages for `raw`. |
| `PACKAGE_NOT_ALLOWED` | `InvalidInputError` | A package is not in the allow-list. |
| `PACKAGE_INSTALL_FAILED` | `InvalidInputError` | The package manager refused the packages. |
| `INSTANCE_INVALID_ID` | `InvalidInputError` | The value breaks the instance id rule. |
| `SCHEDULER_STATE` | `InvalidInputError` | The scheduler is not started, already started, or stopping. |
| `HOST_DEPENDENCY` | `ConfigError` | A binary (QEMU, qemu-img, qemu-storage-daemon, gvproxy-wrapper), system user, cgroup or ACL support is missing or too old, or this user cannot run it; with KVM or HVF, the host CPU cannot run the x86 guest CPU, or the ARM host CPU identity cannot be read. |
| `EXTRA_MISSING` | `ConfigError` | A Python extra is not installed (the message gives the pip command). |
| `SETTING_INVALID` | `ConfigError` | A worker setting is not valid. |
| `ASSET_NOT_FOUND` | `ConfigError` | A VM image is not on this host and was not downloaded. |
| `IMAGE_DAMAGED` | `ConfigError` | A local guest image does not match its digest: download or rebuild the images. |
| `PLATFORM_UNSUPPORTED` | `ConfigError` | The OS is not Linux or macOS. |
| `MIGRATION_UNSUPPORTED` | `ConfigError` | This worker's QEMU cannot save or load VM state (below QEMU 11.0, or a capability refused). |
| `INSTANCE_CONFIG` | `ConfigError` | The instance settings are wrong (store, keyring, store check). |
| `INSTANCE_KEY_MISSING` | `ConfigError` | The instance's key is not in this keyring. The data is intact: do not delete the instance. |
| `INSTANCE_HOST_MISMATCH` | `RouteError` | This worker cannot load the saved instance: a value in `required` differs, or (empty `required`) this release cannot read the stored data or runs other devices for it (`describe_instance(id).host.device_layout`). |
| `CAPACITY_NEVER_FITS` | `RouteError` | The VM is larger than this worker's whole budget. |
| `MIGRATION_REJECTED` | `RouteError` | QEMU refused the saved state. It shows only in the cause chain of `INSTANCE_HOST_MISMATCH` (this QEMU is older than the saver's) or of `InternalError` `INTERNAL` (the same or a newer QEMU). |
| `INSTANCE_NOT_FOUND` | `InstanceGoneError` | No instance with this id (deleted, expired, or never saved). |
| `INSTANCE_EXPIRED` | `InstanceGoneError` | Part of the saved state is gone from the store. |
| `INSTANCE_INTEGRITY` | `InstanceGoneError` | Stored data failed its checks. A head this release authenticates but cannot read gives `RouteError` (`INSTANCE_HOST_MISMATCH`), not this code. |
| `CLEANUP_UNCONFIRMED` | `InternalError` | The end of a QEMU or gvproxy process, or the release of VM resources, was not confirmed. |
| `GUEST_PROTOCOL` | `InternalError` | The guest sent a protocol error or an answer of the wrong type. |
| `INTERNAL` | `InternalError` | A bug. |

## Performance

The start time of a VM depends on its source: [Snapshot cache](#snapshot-cache).

### Warm pool

Set `warm_pool_size` to keep that many VMs ready for each language. Only `VmSpec(language=...)` with no other field comes from the warm pool.

```python
async with Scheduler(SchedulerConfig(warm_pool_size=1)) as scheduler:
    await scheduler.wait_pool_ready()  # Waits until the pool is full, 120 s at most
    async with await scheduler.new(VmSpec(language=Language.PYTHON)) as session:  # From the warm pool
        ...
```

Each warm-pool VM takes 272 MB of the memory budget (192 MB + 80 MB overhead): `warm_pool_size=5` takes about 4.1 GB (5 VMs × 3 languages).

### Concurrent throughput

The default overcommit ratios come from tuning on production KVM hosts. To serve more requests at once, add hosts; do not raise the ratios. On a desktop, find the best values with `make bench-optimizer` in a clone of the repository ([Contributing](#contributing)).

## Caches and disk use

### Where files go

The cache folder (`<cache dir>`) is `~/Library/Caches/exec-sandbox/` on macOS and `~/.cache/exec-sandbox/` on Linux (or `$XDG_CACHE_HOME/exec-sandbox/` when that is an absolute path). It holds:

- `v<version>/`: the download cache (VM images and gvproxy-wrapper)
- `memory-snapshots/` (L1) and `disk-snapshots/` (L2)
- `home-disks/`: the home disks of running VMs
- `instance-assets/`: guest images from the instance store (removed oldest first when the home disks need the space, never one a VM runs on)

`EXEC_SANDBOX_CACHE_DIR` moves all of them (the download cache is then the folder itself, with no `v<version>`). Use one cache folder per host. Put the home disk folder and the L1 cache on a disk, not on tmpfs (`/tmp` on many Linux hosts): on tmpfs, QEMU writes count against the memory of the VM.

### Snapshot cache

A new VM comes from the first source that has it (times with KVM or HVF):

| Source | Start time |
|--------|------------|
| Warm pool | Under 1 ms |
| L1 memory snapshot (a spec that ran before; QEMU 11.0 or later) | About 50-60 ms |
| L2 disk snapshot (packages already installed) | About 200 ms boot, then 0.1-1 s REPL start |
| L3 snapshot cache (L2 from S3) | Download, then as L2 |
| Cold boot | As L2, plus the package install |

After a boot, the REPL starts in the first `exec()` (counted in `timing.execute_ms`, not in `startup_timing`).

#### L3: S3 snapshot cache

L3 shares L2 entries between hosts through S3. Set `s3_bucket` to turn it on (keys under `s3_snapshot_prefix`). It needs `s3:GetObject`, `s3:PutObject`, `s3:PutObjectTagging` and `s3:AbortMultipartUpload` on `<s3_snapshot_prefix>*`. Add two lifecycle rules on that prefix: one that expires objects (uploads have the tag `ttl_days=14`, `EXEC_SANDBOX_SNAPSHOT_CACHE_TTL_DAYS`), and `AbortIncompleteMultipartUpload` (1 day). An L3 error only logs a warning.

### Clean up

L1 and L2 have no size limit. exec-sandbox removes entries (oldest last use first) only when the disk is full or a new home disk needs the space, and never an entry that a running VM uses. When no `Scheduler` runs, you can delete L1 and L2 entries, old `v<version>` image folders and the files in `instance-assets`.

## VM images and offline use

exec-sandbox needs VM images (kernel, initramfs, qcow2 base images) and the gvproxy-wrapper binary. It downloads them from GitHub Releases at first use.

### How images download

Each `Scheduler` start uses the first of these folders that holds both the kernel and the initramfs:

1. `images_dir` ([SchedulerConfig](#schedulerconfig))
2. `EXEC_SANDBOX_IMAGES_DIR`
3. `images/dist/` of a clone of the repository (a local build)
4. the download cache ([Where files go](#where-files-go))

If no folder holds them, exec-sandbox downloads them to the download cache (with `auto_download_assets=False`: `ConfigError` `ASSET_NOT_FOUND`). It looks for gvproxy-wrapper in the folder of the boot files, then in `gvproxy-wrapper/bin/` of a clone, then in the folders above. It looks for the base image of a language, at the first VM of that language, in the folder of the boot files, then in the download cache. With downloads on, a missing asset downloads to the download cache.

Downloads come from the GitHub release of this version (`v{__version__}`) over HTTPS, and each is checked against its SHA-256 digest from the GitHub API. Set `GITHUB_TOKEN` for a higher GitHub API rate limit.

### Offline hosts and Docker

`sbx prefetch` downloads all assets before use:

```bash
sbx prefetch                    # All assets for this architecture
sbx prefetch --arch aarch64     # Assets of another architecture
sbx prefetch -q                 # No progress output (CI, Docker)
```

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm
# The image also needs QEMU (see Install)
RUN uv pip install --system "exec-sandbox[cli]"
RUN sbx prefetch -q
ENV EXEC_SANDBOX_OFFLINE=1
# A spec with packages still needs PyPI or npm for its first build (L2 cache miss).
# Run the container with --device /dev/kvm for hardware acceleration.
```

### Build images from source

From a clone of the repository, `./scripts/build-images.sh` writes `python-3.14-base-{arch}.qcow2`, `node-1.4-base-{arch}.qcow2` and `raw-base-{arch}.qcow2` to `./images/dist/`.

## Linux host setup

### Run QEMU as qemu-vm

For more isolation on Linux, exec-sandbox can run QEMU as an unprivileged `qemu-vm` user:

```bash
# Create qemu-vm system user, add it to the kvm group (hardware acceleration), and add your user to its group (socket access)
sudo useradd --system --no-create-home --shell /usr/sbin/nologin qemu-vm
sudo usermod -aG kvm qemu-vm
sudo usermod -aG qemu-vm $USER

# Let your user run commands as qemu-vm, and chown and rm as root, with no password
echo "$USER ALL=(qemu-vm) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/qemu-vm
echo "$USER ALL=(root) NOPASSWD: $(command -v chown), $(command -v rm)" | sudo tee -a /etc/sudoers.d/qemu-vm
sudo chmod 0440 /etc/sudoers.d/qemu-vm

# setfacl gives qemu-vm access to the home disks and the L1 cache files
sudo apt install acl

# Log in again, or start a shell with the new group (for a service: sg qemu-vm -c '<command>')
newgrp qemu-vm

# Check: each line must print ok
sudo -n -k -u qemu-vm /bin/sh -c 'umask 007 && exec "$@"' qemu-vm true && echo ok
id -nG | grep -qw qemu-vm && echo ok
sudo -n -k -l chown >/dev/null && sudo -n -k -l rm >/dev/null && echo ok
# the group's members, and the accounts whose primary group it is: only qemu-vm and you
gid=$(getent group qemu-vm | cut -d: -f3); { getent group qemu-vm | cut -d: -f4 | tr , '\n'; awk -F: -v g="$gid" '$4 == g {print $1}' /etc/passwd; } | grep -vx -e '' -e qemu-vm -e "$(id -un)" | grep -q . || echo ok
# in the shell that starts the service, after the cgroup setup below: makes a VM-like cgroup and moves a process into it
d=/sys/fs/cgroup/code-exec/check-$$; mkdir "$d" && sh -c 'echo 0 > "$1/cgroup.procs"' sh "$d" 2>/dev/null && test -e "$d/cgroup.kill" && echo ok; rmdir "$d"
```

- Give these sudo rights to the service account only: `chown` and `rm` as root reach every file. A `(qemu-vm)` rule limited to some commands must allow `/bin/sh`.
- The `qemu-vm` group must have no members other than `qemu-vm` and your user.
- The service must run in the `qemu-vm` group (or as root), inside the cgroup v2 subtree of [Resource limits (cgroups)](#resource-limits-cgroups), on Linux 5.14 or later.
- exec-sandbox runs `chmod a+x` on the folders above the images, the VM files and the L1 cache files (for example your home folder).

Each process checks this setup at its first `Scheduler` start. If a check fails, its VMs run QEMU as your user, and a warning names the failed check.

### Resource limits (cgroups)

cgroup limits (layer 6) need a cgroup v2 subtree that your user owns, and your service must start inside it. A reboot removes this setup:

```bash
sudo mkdir -p /sys/fs/cgroup/code-exec/service
echo "+memory +cpu +pids" | sudo tee /sys/fs/cgroup/cgroup.subtree_control
echo "+memory +cpu +pids" | sudo tee /sys/fs/cgroup/code-exec/cgroup.subtree_control
sudo chown -R "$USER": /sys/fs/cgroup/code-exec
echo $$ | sudo tee /sys/fs/cgroup/code-exec/service/cgroup.procs   # then start the service from this shell
```

Do not name your service cgroup `exec-sandbox`, `exec-sandbox-probe`, `exec-sandbox-watcher`, `warm-pool` or `l1-cache`: exec-sandbox uses these names. Without this setup, exec-sandbox logs "cgroup v2 unavailable, resource limits disabled" and uses `ulimit` only.

### Namespaces

Namespaces (layer 7) need `unshare` from util-linux 2.32 or later. Check: `/usr/bin/unshare --pid --fork --kill-child=SIGKILL -- /usr/bin/true && echo ok`. Without it, exec-sandbox logs "unshare unavailable (namespace isolation disabled)" and starts QEMU without namespaces.

`setpriv --pdeathsig` (util-linux 2.33 or later) makes gvproxy, qemu-storage-daemon and a QEMU that runs as your user end with the service. Check: `setpriv --pdeathsig KILL -- /bin/sh -c : && echo ok`. Without it, exec-sandbox logs "setpriv --pdeathsig unavailable" and starts these processes more slowly when the service uses much memory.
## Instance store operations

### Set up the S3 bucket

`<prefix>` below is `s3_instance_prefix` (default `instances/`).

- **Conditional writes.** The store must support `If-None-Match` and `If-Match` on PUT: AWS S3, Cloudflare R2, MinIO and Ceph RGW 19.2.7 or later do. The `Scheduler` checks this at start.
- **Credentials.** The standard AWS chain (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, `AWS_PROFILE`, `~/.aws/credentials`, or the machine role). For R2, MinIO or Ceph RGW, set `s3_endpoint_url`.
- **L3 cache.** `s3_bucket` also turns on the L3 snapshot cache in the same bucket (keys under `s3_snapshot_prefix`): give the workers its permissions too ([L3: S3 snapshot cache](#l3-s3-snapshot-cache)), else each new L2 entry logs a warning.
- **Permissions.** `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject` and `s3:AbortMultipartUpload` on `<prefix>*`, and `s3:ListBucket` on the bucket (without it, S3 answers 403 for a missing key). An access error raises `ConfigError` (`INSTANCE_CONFIG`). Example for `instances/`:

  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"], "Resource": "arn:aws:s3:::my-bucket/instances/*"},
      {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::my-bucket"}
    ]
  }
  ```

- **Lifecycle rules.** `AbortIncompleteMultipartUpload` (1 day) on `<prefix>`, and an expiration of 1 day on `<prefix>probe/`. To remove unused instances, add an expiration of X days on `<prefix>s/` ([Expiration](#expiration)); without it, instances stay until `delete_instance()`. Never expire `<prefix>assets/`: all instances share these guest images. Delete an image there only when no instance uses it (`instance_admin`: `list_assets()`, `delete_asset(digest)`; `describe(id).guest` names the images of an instance whose `access` is `"full"`: keep every image while one is not). A save stores its images before its head: keep the images whose `LastModified` in the bucket is newer than your longest save.
- **No versioning.** With versioning, deleted instances still use storage. If versioning was on, suspend it and add a lifecycle rule that deletes noncurrent versions.
- **No archive storage class** (such as S3 Glacier Flexible Retrieval or Deep Archive): a load cannot read an archived object and fails with `ConfigError` (`INSTANCE_CONFIG`).
- `s3_instance_prefix` and `s3_snapshot_prefix` must not overlap.
- Give write access to the workers only: anyone who can write to the bucket can put back an older head and so roll an instance back.

### Expiration

An expiration rule of X days on `<prefix>s/` deletes each object X days after its write (S3 can take 1-2 more days). Each save writes the head again, and a full copy or a pack (the blocks that changed). A new full copy comes when the packs since the last one reach its size, after 64 packs, or after a key rotation. An instance stays loadable for at least X days after its **last full copy**.

So an instance saved at each request must be used at least every **X / (n + 1) days**, where n = min(full copy size / typical pack size, 64). Example: X = 30 days, a 45 MiB full copy and 1 MiB packs give n = 45: use the instance at least every 15 hours.

When an object is gone, `reuse()` fails with `INSTANCE_EXPIRED` (`action == "RECREATE"`). The next save of a session that is still open repairs the stored instance.

### Key rotation

An instance moves to the newest key at its next save. To rotate the key on all workers:

1. Add the new key at the end of `instance_keyring` on every worker (`OLD,NEW`).
2. When every worker has it, move it to the front (`NEW,OLD`). If one worker puts it first too early, the others fail to load the instances it saves (`INSTANCE_KEY_MISSING`).
3. Remove the old key when no instance uses it: for each id of `list_instances()`, `describe_instance(id).key_id` is not the old key's id. An instance that is never saved again keeps the old key. `sbx instance keys` (or `describe_worker().key_ids`) shows the id of each key of a keyring, newest first.

### Upgrades

A saved instance loads on other exec-sandbox releases and on the same or a newer QEMU: it keeps the QEMU machine type of its first boot (`describe_instance(id).host.machine_type`). On x86 that type (`microvm`) has no version: give x86 workers the QEMU version of the save. A release that changes the devices of the VM or their migration capabilities (`describe_instance(id).host.device_layout`) cannot load the instances of an earlier one: `RouteError` with an empty `required`, before any download. Data that a release cannot read (another release changed the stored format) gives `RouteError` (`INSTANCE_HOST_MISMATCH`) with an empty `required`: load it on a worker of another release. `describe_instance(id).writer_release` names the release of the last save. Old L1 snapshots and VM images stay on disk ([Clean up](#clean-up)).

### Storage layout and costs

- **Store.** Each instance holds its full copy and its packs (at most about twice the full copy) under `<prefix>s/<instance_id>/`. The store holds one copy of each guest image under `<prefix>assets/`.
- **Encryption.** The RAM, the home disk, the `VmSpec` and the host values use AES-256-GCM. The key names, instance id, generation, commit ids and key id are in clear. Guest images are not encrypted.
- **Worker disk.** Each open session of a saved instance, and each save and load, keeps a RAM file and a home disk copy (not encrypted) in a private folder (mode 0700) under the home disk folder (`EXEC_SANDBOX_HOME_DISK_DIR`, default `<cache dir>/home-disks/`). exec-sandbox deletes them when the session closes, or at the next start after a crash. Guest images from the store stay in `<cache dir>/instance-assets/` ([Clean up](#clean-up)).

## CLI reference

### sbx run options

`sbx run --help` prints the same list.

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--code` | `-c` | Code to run (repeatable, in place of SOURCES) | - |
| `--language` | `-l` | python, javascript or raw | from the file extension, else python |
| `--timeout` | `-t` | Timeout in seconds (1-300) | 30 |
| `--env` | `-e` | Environment variable KEY=VALUE (repeatable) | - |
| `--memory` | `-m` | Memory in MB (128 or more) | 192 |
| `--package` | `-p` | Package to install (repeatable) | - |
| `--no-validation` | | Skip the package allow-list check | false |
| `--network` | | Outbound network. Without `--allow-domain`: python reaches PyPI only, javascript npm only, raw any host | false |
| `--allow-domain` | | Allowed domain and its subdomains (repeatable; needs `--network`) | - |
| `--expose` | | Expose port: `INTERNAL[:EXTERNAL][/tcp\|udp]` (repeatable) | - |
| `--upload` | | Upload file `LOCAL:GUEST` (repeatable) | - |
| `--download` | | Download file `GUEST:LOCAL`, or `GUEST` to `./<file name>` (repeatable) | - |
| `--instance` | | Saved instance to continue (made at the first run), saved after the code runs ([Instance ids](#instance-ids)) | - |
| `--json` | | JSON output ([JSON output](#json-output)) | false |
| `--quiet` | `-q` | Hide progress output | false |
| `--debug` | | DEBUG logging, and kernel and init boot logs on stderr (skips the warm pool and L1) | false |

### Exit codes

| Code | When |
|------|------|
| 0-255 | The exit code of the code ([Exit codes and timeouts](#exit-codes-and-timeouts)) |
| 2 | Usage error (also `--allow-domain` without `--network`, or an invalid VM option such as `-m 100` or more than 10 `--expose`), or an error with action `FIX_INPUT` or `FIX_CONFIG` |
| 124 | The code timed out (with `--json`, `result.exit_code` stays -1), or the VM did not boot in time (`VM_BOOT_TIMEOUT`) |
| 125 | Other sandbox and instance errors |
| 130 | Ctrl+C (a second Ctrl+C stops the VMs at once) |

With several sources, the run exits with 2 if a source has a `FIX_INPUT` or `FIX_CONFIG` error, else 124 if a source timed out or did not boot in time, else the highest exit code of the sources. When the code itself exits with 2 or 124, that is only its exit code. In the JSON array, the `exit_code` of a source that timed out is 124.

### JSON output

| Sources | `--json` prints | Fields |
|---------|-----------------|--------|
| One | One object | `result` (`stdout`, `stderr`, `exit_code` and the timings), `startup`, `exposed_ports`, `instance` and `total_ms` |
| Several | An array of objects | `index`, `source`, `run` (the object above, or `null`), `error`, `code`, `action` and `exit_code` |

A single source that fails before its code runs prints no JSON: the error goes to stderr.

### Other commands

| Command | Use | Exit codes |
|---------|-----|------------|
| `sbx instance list\|describe\|keys\|rm` | [Use instances from the command line](#use-instances-from-the-command-line) | 0 on success; 2 for invalid settings, a bad id, or a `FIX_INPUT` or `FIX_CONFIG` error; 125 for the other errors; 130 after Ctrl+C |
| `sbx instance check` | [Use instances from the command line](#use-instances-from-the-command-line) | 0 when every instance passes; 1 when one fails; otherwise as `sbx instance list` |
| `sbx worker describe` | [Use instances from the command line](#use-instances-from-the-command-line) | As `sbx instance list` |
| `sbx prefetch` | [Offline hosts and Docker](#offline-hosts-and-docker) | 1 when a download fails |

`sbx ARGS...` is `sbx run ARGS...`. `sbx -V` shows the version. `exec-sandbox` is the same command as `sbx`.

## API reference

The docstring of each method (`help(Scheduler.new)`) lists its errors.

### Scheduler

| Member | Description |
|--------|-------------|
| `Scheduler(config=None)` | Use as `async with Scheduler(config) as scheduler:`; `None` uses the defaults ([SchedulerConfig](#schedulerconfig)) |
| `new(spec, *, idle_timeout_seconds=None, on_boot_log=None, instance_id=None)` | Start a new VM; returns its `Session` ([Sessions](#sessions)) |
| `reuse(instance_id, *, idle_timeout_seconds=None)` | Start a VM from a saved instance; returns the `Session` that continues it ([Saved instances](#saved-instances)) |
| `wait_pool_ready(timeout=120)` | Wait until the warm pool is full, or until `timeout` seconds pass ([Warm pool](#warm-pool)) |
| `describe_instance(instance_id)` | `InstanceInfo` of a saved instance ([List, describe and delete instances](#list-describe-and-delete-instances)) |
| `list_instances()` | Async iterator of the instance ids in the store |
| `delete_instance(instance_id)` | Delete a saved instance; `True` if it existed |
| `describe_worker()` | `WorkerFacts` of this worker: what a call that raised `RouteError` needs; `facts.satisfies(error.required)` says whether this worker can serve it |
| `config` | The `SchedulerConfig` as given in code: a value set by an `EXEC_SANDBOX_*` variable only is not in it (the `Scheduler` uses it) |

`instance_admin(config)` opens the instance store alone: `async with instance_admin(config) as admin:`, then `admin.describe(instance_id)`, `admin.list_ids()` and `admin.delete(instance_id)`, as the three `Scheduler` methods above ([List, describe and delete instances](#list-describe-and-delete-instances)).

### Session

| Member | Description |
|--------|-------------|
| `exec(code, *, timeout_seconds=None, env_vars=None, on_stdout=None, on_stderr=None)` | Run code; returns an `ExecutionResult`. `timeout_seconds` defaults to `default_timeout_seconds`; `env_vars` is a dict ([Exit codes and timeouts](#exit-codes-and-timeouts)) |
| `write_file(path, content, *, make_executable=False)` | Write a file from bytes, a local `Path` or an `IO[bytes]` stream ([Files](#files)) |
| `read_file(path, *, destination)` | Read a file to a local `Path` or an `IO[bytes]` stream |
| `list_files(path="")` | List a folder; returns `FileInfo` entries (`name`, `is_dir`, `size` in bytes, 0 for folders) |
| `save(*, instance_id=None)` | Save the VM state; returns an `InstanceRef`. `instance_id`: save under a new id and continue it ([Save under a new id](#save-under-a-new-id)) |
| `close()` | Destroy the VM; it never saves ([Rules for save and reuse](#rules-for-save-and-reuse)) |
| `spec` | `VmSpec`: what the VM is made of |
| `startup_timing` | `StartupTiming`: how the VM started (below) |
| `exposed_ports` | `list[ExposedPort]`: `.internal`, `.external`, `.host`, `.protocol`, `.url` |
| `exec_count` | `int`: number of `exec()` calls that returned a result |
| `closed` | `bool`: the session is closed |
| `vm_id` | `str`: id of the VM |
| `instance` | `InstanceRef \| None`: the saved instance this session continues, set by `reuse()` and by each `save()` |

### ExecutionResult

| Field | Type | Description |
|-------|------|-------------|
| `stdout` | str | Output (at most 1 MB) |
| `stderr` | str | Errors (at most 100 KB) |
| `exit_code` | int | 0-255 (0 = success, 128+N = usually killed by signal N), or -1 for a timeout |
| `execution_time_ms` | int \| None | Run time that the VM reports |
| `external_cpu_time_ms` | int \| None | CPU time that the host measures |
| `external_memory_peak_mb` | int \| None | Peak memory that the host measures |
| `external_cpu_nr_throttled` | int \| None | CFS throttle events (host cgroup) |
| `timing.execute_ms` | int | Connect + run + response, measured on the host |
| `timing.connect_ms` | int \| None | Connection to the guest agent (None when not measured) |
| `spawn_ms` | int \| None | Time to get the REPL process (guest) |
| `process_ms` | int \| None | Time from spawn to process exit (guest) |

### StartupTiming

Times in ms of the `new()` or `reuse()` call, measured on the host. After a cold boot, the interpreter starts in the first `exec()` and counts in its `timing.execute_ms`.

| Field | Type | Description |
|-------|------|-------------|
| `total_ms` | int | The whole call |
| `admission_ms` | int | Wait for a resource slot |
| `setup_ms` | int | Cache lookup, overlay, cgroup; after `reuse()`, the restore of the saved state |
| `boot_ms` | int | Cold boot only |
| `warm_pool_hit` | bool | The VM came from the warm pool |
| `l1_cache_hit` | bool | The VM came from an L1 memory snapshot |
| `l1_restore_ms` | int \| None | L1 restore (None unless `l1_cache_hit`) |
| `instance_load_ms` | int \| None | `reuse()` only: read, check and download of the saved state |
| `overlay_ms`, `qemu_cmd_build_ms`, `gvproxy_start_ms`, `qemu_fork_ms`, `guest_wait_ms`, `boot_retries` | int | Parts of a cold boot |

### InstanceRef and InstanceInfo

`save()` returns an `InstanceRef` (`instance_id`, `generation`); pass the id to `reuse()`. `describe_instance()` returns an `InstanceInfo`:

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | str | The id given to `new(instance_id=...)` or `save(instance_id=...)`, else the one the first save made (32 lowercase hex characters) |
| `generation` | int | 1 after the first save; +1 for each later save |
| `recent_commits` | tuple[RecentCommit, ...] | The last saves (`generation`, `commit_id`), newest first, at most 16 |
| `key_id` | str | Id of the keyring key that protects the instance |
| `access` | `"full"`, `"other_release"` or `"no_key"` | `"full"`: this keyring opened the head and this release read it. `"other_release"`: the head opened, but another release wrote it and this release cannot read it. `"no_key"`: this keyring does not have the key, and the other fields are not checked |
| `writer_release` | str \| None | The exec-sandbox release of the last save; None when `access` is `"no_key"`, or when the body names no release |
| `spec` | VmSpec \| None | What the VM is made of (None unless `access` is `"full"`) |
| `host` | HostFacts \| None | The host of the last save: `arch`, `accelerator`, `cpu_model`, `cpu_id`, `qemu_version`, `machine_type`, `release`, `device_layout` (one line for each part of the VM the guest sees: the board, the RAM, each drive and device; then the migration capabilities) (None unless `access` is `"full"`) |
| `guest` | GuestFacts \| None | What the guest got at its first boot: the SHA-256 digests of its files, `base_image`, `packages_image` (None without packages), `kernel`, `initramfs`, and `counter_hz`, the rate of its clock's counter (None when that host had no exact rate); every save carries them (None unless `access` is `"full"`) |
| `egress_domains` | tuple[str, ...] \| None | The domains the VM may reach with the network on, as its last save resolved them; each reuse applies them (empty: all; None unless `access` is `"full"`) |

## Limits

| Resource | Limit |
|----------|-------|
| Code size | 1 MiB |
| stdout / stderr | 1 MB / 100 KB |
| Packages | 50 (214 bytes each) |
| Env vars | 100 (name: 256 bytes, value: 4096 bytes; no control characters except tab) |
| Exposed ports | 10 |
| File I/O | 500 MiB per file; path 4096 bytes (255 per component); 120 s wait for a guest answer (the end of a write, each chunk of a read, a listing) |
| Timeouts | Execution 1-300 s; package install 300 s per try; VM boot 45 s per try, up to 4 tries |
| VM memory | 128 MB minimum (no upper bound) |
| Processes and threads of your code (user 1000) | 1024 (`RLIMIT_NPROC`) |
| Concurrent VMs | Set by admission (host memory, CPU and disk: [SchedulerConfig](#schedulerconfig), [Disk admission](#disk-admission)). A wait of 120 s fails with `TransientError` (`CAPACITY`); a home disk that cannot fit fails at once with `TransientError` (`LOCAL_DISK`) |
| One saved-state object in S3 | About 78 GiB |

## Guest environment

Pre-built images from [GitHub Releases](https://github.com/dualeai/exec-sandbox/releases), all based on Alpine Linux (musl libc):

| Image | Runtime | Package manager | Size |
|-------|---------|-----------------|------|
| `python-3.14-base` | Python 3.14 ([python-build-standalone](https://github.com/astral-sh/python-build-standalone), musl), cloudpickle 3.1.2 | uv 0.13 | ~75 MB |
| `node-1.4-base` | Bun 1.4 | bun | ~71 MB |
| `raw-base` | GNU Bash | None | ~17 MB |

All images include `bash`; the GNU `coreutils`, `grep` (PCRE with `-P`), `findutils`, `sed`, `gawk` and `diffutils`; and `git`, `curl`, `jq`, `openssl`, `patch`, `less`, `make`, `tree`, `tar`, `gzip`, `unzip` and `file`.

- **Python.** Use `uv pip install`: there is no `pip` command. There is no gcc or musl-dev, so packages that need a source build do not install ([Packages](#packages)). `multiprocessing.Pool` works with no setup, also with functions, lambdas and closures defined in the REPL. With the default `cpu_cores=1`, CPU-bound work does not run faster: set `VmSpec(cpu_cores=N)` or use several sessions.
- **JavaScript.** Bun is a Node.js-compatible runtime, not Node.js. `exec()` runs JavaScript and TypeScript with no build step, but not JSX: put JSX in a `.jsx` or `.tsx` file and run it with `bun run`. Use `bun install` for packages and `bun run` for scripts.
- **Shell (raw).** The common tools only, for shell scripts and runtimes that you install yourself. `Language.RAW` runs GNU Bash. Code that starts with `#!` runs as a script file in its own process: its variables and working folder do not stay for the next call.

### Memory and file systems

`memory_mb` holds the kernel (about 20 MB), your processes and the tmpfs mounts. Files in tmpfs use RAM: 10 MB of files use about 10 MB. Each tmpfs mount has a quota per user (UID).

| Mount | Size | Purpose |
|---|---|---|
| `/home/user` | `VmSpec.disk_mb` (host disk, not RAM) | Writable home folder (ext4, nosuid, nodev) |
| `/tmp` | 50% of RAM | Temp files, package manager scratch space |
| `/dev/shm` | 40% of RAM | POSIX shared memory (Python multiprocessing semaphores) |

Compressed swap (zram, lz4) adds 40% of RAM: code can use about 40% more than `memory_mb` (about 269 MB at 192 MB). The compressed data takes at most 20% of RAM. Use timeouts, not memory limits, to stop code that allocates memory without end.

## How it works

1. The `Scheduler` runs in your Python process. It takes a VM from the warm pool, restores it from a snapshot, or boots it ([Snapshot cache](#snapshot-cache)). It uses KVM on Linux and HVF on macOS, else software emulation (TCG): it also runs on cloud VMs and in containers without nested virtualization.
2. In the VM, tiny-init (PID 1) starts the guest agent. The agent gets the code over virtio-serial, runs it in a REPL as a non-root user, and sends stdout and stderr back while the code runs.
3. The root filesystem is read-only (EROFS). `/home/user` is a writable ext4 disk.

### L1 saves on Linux

- An L1 save keeps the VM RAM in `/dev/shm`. In a container, give `/dev/shm` room for the RAM of each L1 save that runs at the same time (`docker run --shm-size`; the default is 64 MB). Else the save is slower.
- For a service that runs as a regular user (uid 1000 or more), run `loginctl enable-linger <user>`, or set `RemoveIPC=no` in `/etc/systemd/logind.conf`. Else an L1 save can fail when the last login session of that user ends (the next save tries again).

### Disk admission

Home disks are sparse. Each VM reserves the part of its `disk_mb` that it has not written yet, divided by `disk_overcommit_ratio`, and `EXEC_SANDBOX_DISK_FREE_FLOOR_MB` (1024 MB) stays free. Admission counts only the home disks of its own `Scheduler`; of other processes, it sees only the space that they use now.

When space is short, admission removes the L1 and L2 cache entries on the same filesystem that no running VM uses (oldest last use first), then waits for VMs to end. When neither makes room, the request fails at once with `TransientError` (`LOCAL_DISK`).

### Memory density

- VMs restored from one L1 memory snapshot share its RAM pages (copy-on-write).
- Linux: KSM page merging, when the host runs KSM (`echo 1 | sudo tee /sys/kernel/mm/ksm/run`).
- Linux with cgroups v2 and host zswap on: exec-sandbox moves the memory of idle warm-pool VMs to zswap.

## Security

To report a vulnerability, follow the [security policy](./SECURITY.md).

### Isolation layers

| Layer | Technology | Protection |
|-------|------------|------------|
| 1 | Hardware virtualization | CPU isolation enforced by hardware (KVM or HVF; not TCG) |
| 2 | Hardened guest kernel | No modules, no io_uring, no eBPF, slab and memory hardening, about 530 kernel options turned off |
| 3 | Unprivileged QEMU | QEMU runs as your user, or as `qemu-vm` on Linux ([Linux host setup](#linux-host-setup)) |
| 4 | Non-root REPL (UID 1000) | Blocks mount, ptrace, raw sockets, kernel modules |
| 5 | seccomp | Linux, when QEMU supports `-sandbox`: a filter on the QEMU process |
| 6 | cgroups v2 | Linux, with a delegated cgroup ([Linux host setup](#linux-host-setup)): memory, CPU and process limits per VM |
| 7 | Namespaces | Linux, when `unshare` works for your user ([Linux host setup](#linux-host-setup)): PID, network, mount, UTS and IPC namespaces for QEMU |
| 8 | Socket authentication (SO_PEERCRED/LOCAL_PEERCRED) | Checks the identity of the QEMU process |

All VMs of a host run as the same user (layer 3): run workloads that must not share files on separate hosts.

### Guarantees

- **One VM per session.** Sessions never share a VM, and `close()` destroys it.
- **No network by default.** Set `allow_network=True` to turn it on.
- **Domain allow-list.** DNS does not resolve the domains that are not allowed. On port 443, the TLS SNI is checked, also against the DNS answers.
- **Package check.** By default, only the top 10,000 PyPI and npm packages.
- **Different random output in each VM**, also in VMs that start from one memory snapshot or one saved instance: [Random values in copies](#random-values-in-copies).

### Supply chain

exec-sandbox checks each download against its SHA-256 digest ([How images download](#how-images-download)). Release assets have [provenance attestations](https://docs.github.com/en/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds): check a file with `gh attestation verify <file> --repo dualeai/exec-sandbox`. Each [release](https://github.com/dualeai/exec-sandbox/releases) includes an SBOM (CycloneDX and SPDX) of the Python dependencies, not of the VM images.

## Contributing

To contribute, open an issue first to discuss the change.

```bash
make install      # Set up the environment
make test         # Run tests
make lint         # Format and lint
make test-sudo    # Tests that need sudo (Linux: qemu-vm user, cgroups); make test does not run them
```

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

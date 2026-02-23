//! QEMU Guest Agent
//!
//! Lightweight async agent running inside QEMU microVMs.
//! Communicates with host via virtio-serial for:
//! - Package installation (pip, npm)
//! - Code execution via persistent REPL (Python, JavaScript, Shell)
//! - Health checks
//!
//! All code execution uses persistent REPL wrappers (not per-exec processes).
//! Each language has a long-lived interpreter process that receives code via
//! a length-prefixed stdin protocol and signals completion via unique
//! sentinels on stderr (nanosecond timestamp + counter). State (variables, imports, functions) persists across
//! executions within the same REPL instance.
//!
//! Uses tokio for fully async, non-blocking I/O.
//! Communication via dual virtio-serial ports:
//! - /dev/virtio-ports/org.dualeai.cmd (host → guest, read-only)
//! - /dev/virtio-ports/org.dualeai.event (guest → host, write-only)

mod types;
use types::*;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::os::unix::io::{AsRawFd, RawFd};
use std::process::Command as StdCommand;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::unix::AsyncFd;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::Command;
use tokio::sync::{Mutex as TokioMutex, mpsc};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const CMD_PORT_PATH: &str = "/dev/virtio-ports/org.dualeai.cmd";
const EVENT_PORT_PATH: &str = "/dev/virtio-ports/org.dualeai.event";

// Execution limits
const MAX_CODE_SIZE_BYTES: usize = 1_000_000; // 1MB max code size
const MAX_PACKAGE_OUTPUT_BYTES: usize = 50_000; // 50KB max package install output
const MAX_TIMEOUT_SECONDS: u64 = 300; // 5 minutes max execution timeout

// Sandbox user identity — all code execution runs as this non-root user.
// Matches the "user" account created in the VM image (see build-qcow2.sh).
const SANDBOX_UID: u32 = 1000;
const SANDBOX_GID: u32 = 1000;

// Connection limits
const MAX_REQUEST_SIZE_BYTES: usize = 16_000_000; // 16MB max single request JSON (streaming chunks are ~90KB each)
const RETRY_DELAY_MS: u64 = 50; // 50ms retry delay on transient errors
// Bounded channel size for the guest→host write queue.
// Larger values let the guest buffer more outbound messages before backpressure
// stalls producers (e.g. file-read chunk generation), improving throughput
// for bursty workloads. The cost is memory: each slot holds a JSON-encoded
// message (~175 KB worst case for file chunks), so 128 slots ≈ 22 MB max.
// Must stay bounded to prevent unbounded memory growth on slow drains.
const WRITE_QUEUE_SIZE: usize = 128;
const READ_TIMEOUT_MS: u64 = 12000; // Timeout for idle reads - detects hung connections
// 12s > 10s health check interval to avoid spurious reconnects

// Host disconnection backoff configuration
// When the host disconnects from virtio-serial, the kernel returns EPOLLHUP immediately on poll().
// Without backoff, the agent would busy-loop consuming 100% CPU. Exponential backoff ensures
// the CPU can enter idle (WFI) state while waiting for host reconnection.
// Note: Even 1ms sleep allows WFI - the kernel enters idle as soon as no tasks are runnable.
const INITIAL_BACKOFF_MS: u64 = 50; // Start with 50ms delay
const MAX_BACKOFF_MS: u64 = 1000; // Cap at 1 second for quick reconnection detection

// Environment variable limits
const MAX_ENV_VARS: usize = 100; // Max number of environment variables
const MAX_ENV_VAR_NAME_LENGTH: usize = 256; // Max env var name length
const MAX_ENV_VAR_VALUE_LENGTH: usize = 4096; // Max env var value length

// Package limits
const MAX_PACKAGES: usize = 50; // Max number of packages per install
const MAX_PACKAGE_NAME_LENGTH: usize = 214; // Max package name length (PyPI limit)
const PACKAGE_INSTALL_TIMEOUT_SECONDS: u64 = 300; // 5 min timeout for package installs

// Streaming configuration (Jan 2026 best practice)
// - 50ms flush interval for low-latency real-time feel
// - 64KB max buffer to prevent memory exhaustion
// - Backpressure via bounded channel when buffer full
const FLUSH_INTERVAL_MS: u64 = 50; // 50ms flush interval (not 1s - too slow for real-time)
const MAX_BUFFER_SIZE_BYTES: usize = 64 * 1024; // 64KB max buffer before forced flush

// File I/O limits
const MAX_FILE_SIZE_BYTES: usize = 500 * 1024 * 1024; // 500 MiB max file size (must match Python MAX_FILE_SIZE_BYTES)
const MAX_FILE_PATH_LENGTH: usize = 4096; // POSIX PATH_MAX (full relative path)
const MAX_FILE_NAME_BYTES: usize = 255; // POSIX NAME_MAX (single component, in bytes)
const SANDBOX_ROOT: &str = "/home/user"; // Sandbox root directory (tmpfs, session-scoped)
const PYTHON_SITE_PACKAGES: &str = "/usr/lib/python3/site-packages"; // ext4, snapshot-persisted
const NODE_MODULES_SYSTEM: &str = "/usr/local/lib/node_modules"; // ext4, snapshot-persisted

// File transfer streaming
// 128KB balances fewer frames (halves syscalls, JSON parses, base64 en/decodes vs 64KB)
// while staying within virtio queue depth (128-256 descriptors). On-wire size after
// base64+JSON is ~175KB per frame. The kernel virtio-vsock 4KB→64KB patch (v5.4) showed
// the biggest throughput win; 64KB→128KB reduces per-transfer CPU overhead further.
const FILE_TRANSFER_CHUNK_SIZE: usize = 128 * 1024;
const FILE_TRANSFER_ZSTD_LEVEL: i32 = 3;

// Graceful termination configuration
// - First send SIGTERM to allow process to cleanup (Python atexit, temp files, etc.)
// - Wait grace period for process to exit
// - If still running, send SIGKILL to entire process group
const TERM_GRACE_PERIOD_SECONDS: u64 = 5; // 5 seconds grace period before SIGKILL

/// Regex for validating package names with version specifiers (required).
///
/// Pattern: Package name + version operator (required) + version spec
/// - Package name: [a-zA-Z0-9_\-\.]+
/// - Version operator: [@=<>~] (at least one required)
/// - Version spec: [a-zA-Z0-9_\-\.@/=<>~\^\*\[\], ]*
///
/// Supports:
/// - npm: lodash@4.17.21, lodash@~4.17, lodash@^4.0.0
/// - Python: pandas==2.0.0, pandas~=2.0, pandas>=2.0,<3.0
///
/// Rejects packages without version: "pandas", "lodash"
static PACKAGE_NAME_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[a-zA-Z0-9_\-\.]+[@=<>~][a-zA-Z0-9_\-\.@/=<>~\^\*\[\], ]*$").unwrap()
});

/// Blacklist of dangerous environment variables.
///
/// Security rationale:
/// - LD_PRELOAD/LD_LIBRARY_PATH/LD_AUDIT: Arbitrary code execution via library injection
/// - BASH_ENV/ENV: Execute arbitrary file on shell startup
/// - PATH: Executable search path manipulation (could bypass sandboxing)
/// - GCONV_PATH: glibc converter modules (code injection)
/// - HOSTALIASES: DNS resolution manipulation
/// - PROMPT_COMMAND: Execute arbitrary commands in bash
/// - MALLOC_*: Memory allocator hooks (potential exploitation)
/// - NODE_OPTIONS: Node.js runtime options (can execute arbitrary code)
/// - PYTHONWARNINGS/PYTHONSTARTUP: Python module injection
/// - GLIBC_TUNABLES: CVE-2023-4911 buffer overflow
static BLOCKED_ENV_VARS: &[&str] = &[
    // Dynamic linker (arbitrary code execution)
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "LD_BIND_NOW",
    "LD_DEBUG",
    "LD_DEBUG_OUTPUT",
    "LD_USE_LOAD_BIAS",
    "LD_PROFILE",
    "LD_ORIGIN_PATH",
    "LD_AOUT_LIBRARY_PATH",
    "LD_AOUT_PRELOAD",
    // glibc tunables - CVE-2023-4911
    "GLIBC_TUNABLES",
    // Node.js runtime
    "NODE_OPTIONS",
    "NODE_REPL_HISTORY",
    // Python runtime
    "PYTHONWARNINGS",
    "PYTHONSTARTUP",
    "PYTHONHOME",
    // Shell environment execution
    "BASH_ENV",
    "ENV",
    "PROMPT_COMMAND",
    // Path manipulation
    "PATH",
    // glibc/system hooks
    "GCONV_PATH",
    "HOSTALIASES",
    "LOCPATH",
    "NLSPATH",
    "RESOLV_HOST_CONF",
    "RES_OPTIONS",
    "TMPDIR",
    "TZDIR",
    "MALLOC_CHECK_",
    "MALLOC_TRACE",
    "MALLOC_PERTURB_",
];

// =============================================================================
// Persistent REPL Wrappers
// =============================================================================

/// Python REPL wrapper: exec() in persistent namespace with sentinel protocol.
///
/// Protocol:
///   Input:  "{sentinel_id} {code_len}\n" + code_bytes on stdin
///   Output: stdout/stderr from user code, then __SENTINEL_{id}_{exit_code}__\n on stderr
///
/// Uses sys.stdin.buffer (binary mode) to read exact byte counts.
/// Text-mode sys.stdin.read(n) reads n *characters*, which differs from n bytes
/// for multi-byte UTF-8 (e.g., "café" = 5 chars but 6 bytes), causing deadlocks.
const PYTHON_REPL_WRAPPER: &str = r#"import os as _repl_os
import sys
import traceback
import ctypes as _repl_ctypes

# Security: set PR_SET_DUMPABLE=0 to prevent ptrace from other UID 1000 processes.
# Must be done here (after exec) because begin_new_exec() always resets dumpable
# to 1, regardless of credential state. Blocks CVE-2022-30594 style attacks.
_repl_ctypes.CDLL("libc.so.6", use_errno=True).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE=0

sys.argv = ['-c']

# Use __main__.__dict__ as exec namespace so functions have correct __globals__.
# This lets pickle serialize exec()'d functions by qualified name, and cloudpickle
# recognize __main__.__dict__ for minimal globals extraction.
import __main__

ns = __main__.__dict__
ns["__builtins__"] = __builtins__

# Force fork start method — Python 3.14 defaults to forkserver, which hangs in the
# single-process VM environment. fork is safe here (single-threaded, Linux).
import multiprocessing
multiprocessing.set_start_method("fork")

# Lazy-load cloudpickle: only imported when multiprocessing actually spawns a process.
# Saves ~100-150ms on REPL startup for scripts that never use multiprocessing.
# cloudpickle is safe to defer (pure Python, no import-time side effects).
# See: PEP 810 for future native lazy imports (Python 3.15+).
def _patch_cloudpickle():
    import cloudpickle
    import copyreg
    import io
    import multiprocessing.reduction as _reduction

    class _CloudForkingPickler(cloudpickle.Pickler):
        _extra_reducers = {}
        _copyreg_dispatch_table = copyreg.dispatch_table

        def __init__(self, *args, **kwds):
            super().__init__(*args, **kwds)
            self.dispatch_table = self._copyreg_dispatch_table.copy()
            self.dispatch_table.update(self._extra_reducers)

        @classmethod
        def register(cls, type, reduce):
            cls._extra_reducers[type] = reduce

        @classmethod
        def dumps(cls, obj, protocol=None):
            buf = io.BytesIO()
            cls(buf, protocol).dump(obj)
            return buf.getbuffer()

        loads = staticmethod(cloudpickle.loads)

    def _cloud_dump(obj, file, protocol=None):
        _CloudForkingPickler(file, protocol).dump(obj)

    _reduction.ForkingPickler = _CloudForkingPickler
    _reduction.dump = _cloud_dump

# Intercept multiprocessing.Process.start() to trigger lazy patching.
# After first call, restores the original start() to avoid overhead.
_mp_orig_start = multiprocessing.Process.start
_mp_patched = False
def _lazy_mp_start(self):
    global _mp_patched
    if not _mp_patched:
        _mp_patched = True
        _patch_cloudpickle()
        multiprocessing.Process.start = _mp_orig_start
    return _mp_orig_start(self)
multiprocessing.Process.start = _lazy_mp_start

# PID guard: forked children inherit the REPL wrapper. Record parent PID so
# children that escape user code (via sys.exit(), exception, or fall-through)
# flush their output and terminate without writing a premature sentinel.
_repl_pid = _repl_os.getpid()

# Redirect sys.stdin to /dev/null so user code that reads stdin (input(),
# sys.stdin.read(), for line in sys.stdin, etc.) gets immediate EOF instead
# of blocking on the protocol pipe. The REPL loop reads from _stdin_buf
# (the original stdin buffer) for the length-prefixed command protocol.
_stdin_buf = sys.stdin.buffer
sys.stdin = open(_repl_os.devnull, "r")

while True:
    header = _stdin_buf.readline()
    if not header:
        break
    sentinel_id, code_len = header.decode().strip().split(" ", 1)
    code = _stdin_buf.read(int(code_len)).decode()
    exit_code = 0
    try:
        compiled = compile(code, "<exec>", "exec")
        exec(compiled, ns)
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    # PID guard: if we're a forked child, flush output and terminate immediately
    # without writing a sentinel. Only the original REPL parent writes sentinels.
    if _repl_os.getpid() != _repl_pid:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        _repl_os._exit(exit_code)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stderr.write(f"__SENTINEL_{sentinel_id}_{exit_code}__\n")
    sys.stderr.flush()
"#;

/// JavaScript REPL wrapper: vm.runInContext() with sentinel protocol (Bun-compatible).
///
/// Uses Uint8Array byte-level I/O for correct byte counts with multi-byte UTF-8.
/// Same length-prefixed stdin protocol and stderr sentinel as Python wrapper.
/// Uses Bun.Transpiler with replMode for proper REPL semantics: last-expression capture
/// as `{ value: expr }`, const→var hoisting, top-level await via async IIFE wrapping.
/// Awaits the returned value if thenable (handles fire-and-forget async calls like `main()`).
/// Catches unhandled promise rejections via process.on('unhandledRejection').
/// Provides Bun's native `require` for CommonJS package imports (resolves global packages).
/// Provides `__import()` to the VM context — a host-scope wrapper for dynamic `import()`.
/// Bun's replMode transpiler rewrites `import()` → `__import()` automatically; a regex
/// fallback also catches any cases the transpiler misses.
const JS_REPL_WRAPPER: &str = r#"// Security: set PR_SET_DUMPABLE=0 to prevent ptrace from other UID 1000 processes.
// Must be done here (after exec) because begin_new_exec() always resets dumpable
// to 1, regardless of credential state. Blocks CVE-2022-30594 style attacks.
import { dlopen, FFIType } from 'bun:ffi';
try {
    const _libc = dlopen('libc.so.6', { prctl: { args: [FFIType.i32, FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64], returns: FFIType.i32 } });
    _libc.symbols.prctl(4, 0, 0, 0, 0);  // PR_SET_DUMPABLE=0
    _libc.close();
} catch (_) {}

import { createContext, runInContext } from 'node:vm';
import { Readable } from 'node:stream';
// Use 'ts' loader to accept both JavaScript and TypeScript syntax (TS is a
// superset of JS). We avoid 'tsx' because Bun's TSX parser has open bugs with
// generic arrow defaults (<T = any>() => {}, see oven-sh/bun#4985) and
// angle-bracket type assertions are ambiguous with JSX.
const transpiler = new Bun.Transpiler({ loader: 'ts', replMode: true });
// Host-scope dynamic import wrapper. The VM's runInContext cannot use the
// native import() keyword (it's a language feature, not a function). This
// arrow function captures the host ESM module's import() capability, and
// is exposed to the VM context so transpiled code can call __import() instead.
const __import = (specifier) => import(specifier);
// Create a null stdin (immediate EOF) so user code that reads process.stdin
// or Bun.stdin gets EOF instead of blocking on the protocol pipe.
const nullStdin = new Readable({ read() { this.push(null); } });
nullStdin.fd = -1;
nullStdin.isTTY = false;
// Proxy intercepts stdin access on process, forwarding everything else.
// set trap prevents user code from corrupting the real process object.
const sandboxProcess = new Proxy(process, {
    get(target, prop) {
        if (prop === 'stdin') return nullStdin;
        const val = target[prop];
        return typeof val === 'function' ? val.bind(target) : val;
    },
    set(target, prop, value) {
        if (prop === 'stdin') return true;
        target[prop] = value;
        return true;
    }
});
// Also proxy Bun to intercept Bun.stdin — without this, user code could
// bypass the process.stdin redirect via Bun.stdin.stream().
const sandboxBun = new Proxy(Bun, {
    get(target, prop) {
        if (prop === 'stdin') return nullStdin;
        const val = target[prop];
        return typeof val === 'function' ? val.bind(target) : val;
    }
});
const ctx = createContext({
    Bun: sandboxBun,
    console, process: sandboxProcess, setTimeout, setInterval, clearTimeout, clearInterval,
    Buffer, URL, URLSearchParams, TextEncoder, TextDecoder, fetch,
    Request, Response, Headers, Blob, FormData,
    ReadableStream, WritableStream, TransformStream,
    AbortController, AbortSignal, Event, EventTarget,
    WebAssembly,
    atob, btoa, structuredClone, queueMicrotask,
    crypto,
    require,
    __import,
    module: { exports: {} },
    exports: {},
    __filename: '<exec>', __dirname: '/tmp',
});
// Catch unhandled rejections from fire-and-forget promises (e.g. on non-last lines).
// Sets exitCode so the sentinel reports failure.
let unhandledRejection = null;
process.on('unhandledRejection', (reason) => {
    unhandledRejection = reason;
});
const stdin = Bun.stdin.stream();
const reader = stdin.getReader();
let buf = new Uint8Array(0);
const dec = new TextDecoder();
function cat(a, b) {
    const r = new Uint8Array(a.length + b.length);
    r.set(a); r.set(b, a.length);
    return r;
}
async function readLine() {
    while (true) {
        const idx = buf.indexOf(10);
        if (idx !== -1) {
            const line = dec.decode(buf.slice(0, idx));
            buf = buf.slice(idx + 1);
            return line;
        }
        const { done, value } = await reader.read();
        if (done) return null;
        buf = cat(buf, value);
    }
}
async function readN(n) {
    while (buf.length < n) {
        const { done, value } = await reader.read();
        if (done) return null;
        buf = cat(buf, value);
    }
    const data = buf.slice(0, n);
    buf = buf.slice(n);
    return dec.decode(data);
}
while (true) {
    const header = await readLine();
    if (header === null) break;
    const sp = header.indexOf(' ');
    const sentinelId = header.substring(0, sp);
    const codeLen = parseInt(header.substring(sp + 1), 10);
    const code = await readN(codeLen);
    if (code === null) break;
    let exitCode = 0;
    unhandledRejection = null;
    try {
        // Bun.Transpiler with replMode transforms code for REPL semantics:
        // - Wraps in async IIFE for top-level await support
        // - Captures last expression as { value: (expr) }
        // - Converts const/let to var for re-declaration across invocations
        const transformed = transpiler.transformSync(code);
        // Defensive fallback: rewrite any import() the transpiler didn't
        // catch. Bun's replMode already rewrites import() → __import(),
        // so this regex normally never matches.
        // \b prevents false positives on identifiers like "reimport(".
        const patched = transformed.replace(/\bimport\s*\(/g, '__import(');
        if (patched.length > 0) {
            let val = runInContext(patched, ctx, { filename: '<exec>' });
            // replMode wraps in async IIFE only when code has top-level await;
            // otherwise runInContext returns { value: expr } directly.
            if (val && typeof val.then === 'function') {
                val = await val;
            }
            // If the last expression was a Promise (e.g. `main()`),
            // await it so async work completes before the sentinel.
            if (val && val.value && typeof val.value.then === 'function') {
                await val.value;
            }
        }
    } catch (e) {
        process.stderr.write((e && e.stack ? e.stack : String(e)) + '\n');
        exitCode = 1;
    }
    // Check for unhandled rejections from non-last-expression promises
    if (unhandledRejection !== null) {
        const r = unhandledRejection;
        process.stderr.write((r && r.stack ? r.stack : String(r)) + '\n');
        exitCode = 1;
    }
    process.stderr.write(`__SENTINEL_${sentinelId}_${exitCode}__\n`);
}
"#;

/// Shell REPL wrapper: eval with sentinel protocol.
///
/// Shell `read` + `head -c` operate on bytes (not characters), so the
/// length-prefixed protocol is naturally correct for multi-byte UTF-8.
/// `eval "$code"` maintains shell state (env vars, cwd, functions, aliases).
const SHELL_REPL_WRAPPER: &str = r#"while read sentinel_id code_len; do
    code=$(head -c "$code_len")
    eval "$code" < /dev/null
    _ec=$?
    printf "__SENTINEL_%s_%d__\n" "$sentinel_id" "$_ec" >&2
done
"#;

/// Module-level REPL storage. Persists across 12s guest agent reconnect cycles
/// (guest agent process stays alive, just reconnects serial port).
/// Monotonic counter for sentinel IDs. Combined with nanosecond timestamp
/// to produce unique, unpredictable IDs without the uuid crate dependency.
static SENTINEL_COUNTER: AtomicU64 = AtomicU64::new(0);

static REPL_STATES: Lazy<TokioMutex<HashMap<String, ReplState>>> =
    Lazy::new(|| TokioMutex::new(HashMap::new()));

/// Write REPL wrapper script to SANDBOX_ROOT for the given language.
/// All languages use SANDBOX_ROOT so package managers (bun, pip) resolve
/// dependencies installed in the same directory.
async fn write_repl_wrapper(language: Language) -> Result<(), Box<dyn std::error::Error>> {
    match language {
        Language::Python => {
            tokio::fs::write(format!("{SANDBOX_ROOT}/_repl.py"), PYTHON_REPL_WRAPPER).await?
        }
        Language::Javascript => {
            tokio::fs::write(format!("{SANDBOX_ROOT}/_repl.mjs"), JS_REPL_WRAPPER).await?
        }
        Language::Raw => {
            tokio::fs::write(format!("{SANDBOX_ROOT}/_repl.sh"), SHELL_REPL_WRAPPER).await?
        }
    }
    Ok(())
}

/// Spawn a fresh REPL process for the given language.
///
/// Scripts run from SANDBOX_ROOT so package resolution works naturally:
///   - Python: PYTHONPATH includes {SANDBOX_ROOT}/site-packages + {PYTHON_SITE_PACKAGES}
///   - JavaScript: Node resolution finds {SANDBOX_ROOT}/node_modules + {NODE_MODULES_SYSTEM}
///   - Raw: GNU Bash (not busybox ash) for arrays, traps, [[ ]], etc.
async fn spawn_repl(language: Language) -> Result<ReplState, Box<dyn std::error::Error>> {
    write_repl_wrapper(language).await?;

    let mut cmd = match language {
        Language::Python => {
            let mut c = Command::new("python3");
            c.arg(format!("{SANDBOX_ROOT}/_repl.py"));
            // Package resolution: session-time installs (tmpfs) then system installs (ext4 snapshot)
            // See: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH
            c.env(
                "PYTHONPATH",
                format!("{SANDBOX_ROOT}/site-packages:{PYTHON_SITE_PACKAGES}"),
            );
            // Disable the 4300-digit limit for int<->str conversions (CVE-2020-10735).
            // Safe here: execution timeout (300s) and output caps (1MB) already bound resource usage.
            // See: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONINTMAXSTRDIGITS
            c.env("PYTHONINTMAXSTRDIGITS", "0");
            // Prevent Python from prepending script directory or CWD to sys.path,
            // blocking stdlib shadowing attacks (CVE-2024-48990, CVE-2025-68668).
            // See: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONSAFEPATH
            c.env("PYTHONSAFEPATH", "1");
            c
        }
        Language::Javascript => {
            let mut c = Command::new("bun");
            // --smol: configure JSC to use smaller heap that grows slower.
            // Reduces heap from ~343MB to ~54MB at ~2x CPU cost — always
            // worthwhile in 128-256MB sandbox VMs.
            c.arg("--smol");
            c.arg(format!("{SANDBOX_ROOT}/_repl.mjs"));
            // Package resolution: session-time (tmpfs) then system (ext4 snapshot)
            c.env(
                "NODE_PATH",
                format!("{SANDBOX_ROOT}/node_modules:{NODE_MODULES_SYSTEM}"),
            );
            // Disable FTL (most expensive JIT tier). FTL requires 100k+
            // iterations to trigger — never reached by sandbox code snippets.
            // Saves JIT code buffer memory.
            c.env("BUN_JSC_useFTLJIT", "0");
            c
        }
        Language::Raw => {
            // Use bash instead of /bin/sh (busybox ash on Alpine). Ash silently
            // breaks common shell patterns: arrays, process substitution, [[ ]],
            // here-strings, traps. Bash is installed in all VM images (COMMON_PKGS).
            // --norc/--noprofile: skip startup files (defense-in-depth; BASH_ENV
            // and PROMPT_COMMAND are already in the env var blocklist).
            //
            // NOTE: Shell REPL has dumpable=1 (cannot set prctl from bash after
            // exec). Python and JS wrappers set dumpable=0 in their wrapper code.
            let mut c = Command::new("bash");
            c.args(["--norc", "--noprofile"]);
            c.arg(format!("{SANDBOX_ROOT}/_repl.sh"));
            c
        }
    };

    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .current_dir(SANDBOX_ROOT)
        .process_group(0);
    // NOTE: .uid()/.gid() intentionally NOT used — see pre_exec below.

    // Defense-in-depth: harden the REPL child process before exec.
    //
    // We handle uid/gid/prctl ALL in our pre_exec to control ordering.
    // Rust's Command applies .uid()/.gid() in its own pre_exec which runs
    // AFTER ours. The UID change from root to SANDBOX_UID resets PR_SET_DUMPABLE
    // (Linux kernel behavior: commit_creds() resets dumpable on credential
    // change). By doing uid/gid changes ourselves, we can set DUMPABLE=0
    // AFTER the change, protecting the fork→exec window.
    //
    // IMPORTANT: begin_new_exec() in the kernel ALWAYS resets dumpable to
    // SUID_DUMP_USER (1) during execve(), regardless of credentials. So step 4
    // only protects the brief window between fork() and execve(). The REPL
    // wrappers (Python/JS) re-apply PR_SET_DUMPABLE=0 after exec() to get
    // persistent protection. Shell REPL cannot do this (bash can't call prctl).
    //
    // Order matters:
    //   1. no_new_privs — blocks SUID/SGID escalation for all future exec()
    //   2. setgid(SANDBOX_GID) — must come before setuid to avoid losing permission
    //   3. setuid(SANDBOX_UID) — drop to non-root (resets dumpable via commit_creds)
    //   4. dumpable=0 — defense-in-depth for fork→exec window only
    //   5. RLIMIT_NPROC=1024 — after spawn(), via prlimit64 from parent (root)
    //
    // SAFETY: pre_exec runs in the forked child before exec. All syscalls
    // used (prctl, setgid, setuid) are async-signal-safe. Step 5 runs in the
    // parent after spawn — not subject to async-signal-safety constraints.
    unsafe {
        cmd.pre_exec(|| {
            // 1. Block privilege escalation via execve (SUID/SGID, file capabilities).
            //    Inherited by all children and subprocesses.
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // 2-3. Drop to non-root sandbox user. This blocks mount(2), ptrace,
            //      raw sockets, and kernel module loading without seccomp.
            if libc::setgid(SANDBOX_GID) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::setuid(SANDBOX_UID) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // 4. Prevent ptrace attach and core dumps from other sandbox user processes.
            //    Must be AFTER setuid — UID change resets dumpable to 1.
            //    CVE-2022-30594: ptrace + PTRACE_O_SUSPEND_SECCOMP bypass.
            if libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let mut child = cmd.spawn()?;

    // 5. Limit processes+threads for the REPL child (thread/fork bomb prevention).
    //    RLIMIT_NPROC counts all tasks (processes + threads via clone()) for the
    //    real UID. The kernel checks this in copy_process() and returns EAGAIN when
    //    exceeded. Python surfaces this as:
    //      RuntimeError: can't start new thread (threading)
    //      OSError: [Errno 11] Resource temporarily unavailable (os.fork)
    //    Called from the guest-agent (root) via prlimit64 on the child PID, NOT in
    //    pre_exec — musl libc's setrlimit is not async-signal-safe (it may use
    //    __synccall internally), which can deadlock in the post-fork/pre-exec window.
    //    Root (CAP_SYS_RESOURCE) can set rlimits on any process.
    //    Value 1024 matches AWS Lambda's 1,024 processes+threads combined limit.
    //    Both soft and hard set to 1024 — user code cannot raise it.
    //    See: https://man7.org/linux/man-pages/man2/prlimit.2.html
    {
        let nproc_limit = libc::rlimit {
            rlim_cur: 1024,
            rlim_max: 1024,
        };
        let pid = match child.id() {
            Some(id) => id as libc::pid_t,
            None => {
                eprintln!("WARNING: REPL child exited before prlimit64(RLIMIT_NPROC)");
                0 // skip prlimit64 — child already gone
            }
        };
        if pid > 0 {
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_prlimit64,
                    pid,
                    libc::RLIMIT_NPROC,
                    &nproc_limit as *const libc::rlimit,
                    std::ptr::null_mut::<libc::rlimit>(),
                )
            };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                eprintln!(
                    "WARNING: prlimit64(RLIMIT_NPROC) failed for pid {}: {}",
                    pid, err
                );
            }
        }
    }

    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    eprintln!("Spawned REPL for language={}", language.as_str());

    Ok(ReplState {
        child,
        stdin,
        stdout,
        stderr,
    })
}

/// Build code prefix that sets environment variables for the given language.
///
/// For RAW mode with shebang lines (`#!`), the user code is written to a temp
/// file so the kernel's `binfmt_script` handles interpreter dispatch. Without
/// this, `eval` treats `#!` as a comment and tries to parse the remaining lines
/// as bash — failing for non-shell interpreters (awk, perl, ruby, etc.).
fn prepend_env_vars(language: Language, code: &str, env_vars: &HashMap<String, String>) -> String {
    // Strip UTF-8 BOM (U+FEFF) — editors and copy-paste can silently prepend it,
    // causing interpreters to fail with unhelpful errors (e.g. Python SyntaxError,
    // bash "$'\357\273\277': command not found").
    let code = code.strip_prefix('\u{FEFF}').unwrap_or(code);

    let mut full_code = String::new();

    if !env_vars.is_empty() {
        match language {
            Language::Python => {
                full_code.push_str("import os as __os__\n");
                for (key, value) in env_vars {
                    // Escape backslashes and single quotes for Python string literal
                    let escaped_key = key.replace('\\', "\\\\").replace('\'', "\\'");
                    let escaped_val = value.replace('\\', "\\\\").replace('\'', "\\'");
                    full_code.push_str(&format!(
                        "__os__.environ['{escaped_key}']='{escaped_val}'\n"
                    ));
                }
                full_code.push_str("del __os__\n");
            }
            Language::Javascript => {
                for (key, value) in env_vars {
                    let escaped_key = key.replace('\\', "\\\\").replace('\'', "\\'");
                    let escaped_val = value.replace('\\', "\\\\").replace('\'', "\\'");
                    full_code.push_str(&format!("process.env['{escaped_key}']='{escaped_val}';\n"));
                }
            }
            Language::Raw => {
                for (key, value) in env_vars {
                    // Shell export syntax requires unquoted identifier keys.
                    // Skip keys that aren't valid POSIX identifiers to avoid broken syntax.
                    if !key
                        .bytes()
                        .next()
                        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
                        || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                    {
                        eprintln!("Skipping invalid shell env var key: {:?}", key);
                        continue;
                    }
                    let escaped_val = value.replace('\'', "'\"'\"'");
                    full_code.push_str(&format!("export {key}='{escaped_val}'\n"));
                }
            }
        }
    }

    // RAW mode shebang: write to temp file for kernel-level interpretation.
    // eval treats #! as a comment, then tries to parse remaining lines as
    // shell syntax — fails for non-shell interpreters (awk, perl, ruby, etc.).
    if language == Language::Raw && code.starts_with("#!") {
        // Pick a heredoc sentinel not present in the code
        let mut sentinel = String::from("_EXEC_SANDBOX_EOF_");
        while code.contains(&sentinel) {
            sentinel.push('X');
        }
        // IMPORTANT: Each `\<newline><spaces>` in the Rust string literal
        // strips the newline AND all leading whitespace on the continuation
        // line. The resulting shell lines have NO leading indentation. This
        // is required: the heredoc closing sentinel must appear at column 0.
        // Do not reformat alignment without verifying generated output.
        full_code.push_str(&format!(
            "_sf=$(mktemp /tmp/exec_XXXXXX) || {{ printf 'exec-sandbox: mktemp failed\\n' >&2; exit 126; }}\n\
             cat > \"$_sf\" <<'{sentinel}'\n\
             {code}\n\
             {sentinel}\n\
             chmod +x \"$_sf\"\n\
             \"$_sf\"\n\
             _sec=$?\n\
             rm -f \"$_sf\"\n\
             ( exit $_sec )\n"
        ));
        return full_code;
    }

    full_code.push_str(code);
    full_code
}

/// Extract exit code following Unix shell conventions.
///
/// - Normal exit: returns the exit code (0-255)
/// - Signal kill: returns 128 + signal_number (e.g., SIGKILL → 137, SIGSEGV → 139)
/// - Neither (impossible on Unix): returns 255 as "unknown error" fallback
fn exit_code_from_status(status: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .or_else(|| status.signal().map(|sig| 128 + sig))
        .unwrap_or(255)
}

/// Serialize a GuestResponse to JSON + newline and queue it for sending.
async fn send_response(
    write_tx: &mpsc::Sender<Vec<u8>>,
    msg: &GuestResponse,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = serde_json::to_vec(msg)?;
    bytes.push(b'\n');
    write_tx
        .send(bytes)
        .await
        .map_err(|_| "Write queue closed")?;
    Ok(())
}

// Helper to send streaming error via queue
async fn send_streaming_error(
    write_tx: &mpsc::Sender<Vec<u8>>,
    message: String,
    error_type: &str,
    op_id: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    send_response(
        write_tx,
        &GuestResponse::Error {
            message,
            error_type: error_type.to_string(),
            op_id: op_id.map(|s| s.to_string()),
            version: Some(VERSION),
        },
    )
    .await
}

// Note: flush_buffers() removed - replaced with spawned task pattern (Nov 2025)

/// Install packages to system paths for snapshot persistence.
///
/// Packages are installed to system directories on the ext4 rootfs (not /home/user
/// tmpfs) so they survive snapshot capture. The rootfs must be mounted read-write
/// (init.rw=1) during snapshot creation.
///   - Python: `uv pip install --target {PYTHON_SITE_PACKAGES}`
///   - JavaScript: `bun add` in /usr/local/lib ({NODE_MODULES_SYSTEM} resolution)
async fn install_packages(
    language: Language,
    packages: &[String],
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    use tokio::time::Duration;

    // Validation: check language supports package installation
    if language == Language::Raw {
        send_streaming_error(
            write_tx,
            format!(
                "Unsupported language '{}' for package installation (supported: python, javascript)",
                language.as_str()
            ),
            "validation_error",
            None,
        )
        .await?;
        return Ok(());
    }

    // Validation: check packages list is not empty
    if packages.is_empty() {
        send_streaming_error(
            write_tx,
            "No packages specified for installation".to_string(),
            "validation_error",
            None,
        )
        .await?;
        return Ok(());
    }

    // Validation: check package count
    if packages.len() > MAX_PACKAGES {
        send_streaming_error(
            write_tx,
            format!(
                "Too many packages: {} (max {})",
                packages.len(),
                MAX_PACKAGES
            ),
            "validation_error",
            None,
        )
        .await?;
        return Ok(());
    }

    // Validation: check for suspicious package names
    for pkg in packages {
        // Check empty
        if pkg.is_empty() {
            send_streaming_error(
                write_tx,
                "Package name cannot be empty".to_string(),
                "validation_error",
                None,
            )
            .await?;
            return Ok(());
        }

        // Check length
        if pkg.len() > MAX_PACKAGE_NAME_LENGTH {
            send_streaming_error(
                write_tx,
                format!(
                    "Package name too long: {} bytes (max {})",
                    pkg.len(),
                    MAX_PACKAGE_NAME_LENGTH
                ),
                "validation_error",
                None,
            )
            .await?;
            return Ok(());
        }

        // Check for path traversal and suspicious characters
        if pkg.contains("..") || pkg.contains("/") || pkg.contains("\\") {
            send_streaming_error(
                write_tx,
                format!(
                    "Invalid package name: '{}' (path characters not allowed)",
                    pkg
                ),
                "validation_error",
                None,
            )
            .await?;
            return Ok(());
        }

        // Check for null bytes
        if pkg.contains('\0') {
            send_streaming_error(
                write_tx,
                "Package name contains null byte".to_string(),
                "validation_error",
                None,
            )
            .await?;
            return Ok(());
        }

        // Check for control characters
        if pkg.chars().any(|c| c.is_control()) {
            send_streaming_error(
                write_tx,
                format!(
                    "Invalid package name: '{}' (control characters not allowed)",
                    pkg
                ),
                "validation_error",
                None,
            )
            .await?;
            return Ok(());
        }

        // Check against regex (allows alphanumeric, dash, underscore, dot, @, /, =, <, >, ~, [, ])
        if !PACKAGE_NAME_REGEX.is_match(pkg) {
            send_streaming_error(
                write_tx,
                format!(
                    "Invalid package name: '{}' (contains invalid characters)",
                    pkg
                ),
                "validation_error",
                None,
            )
            .await?;
            return Ok(());
        }
    }

    let start = Instant::now();

    let mut cmd = match language {
        Language::Python => {
            let mut c = Command::new("uv");
            c.arg("pip")
                .arg("install")
                .arg("--python")
                .arg("/opt/python/bin/python3")
                .arg("--target")
                .arg(PYTHON_SITE_PACKAGES);
            for pkg in packages {
                c.arg(pkg);
            }
            c.current_dir(PYTHON_SITE_PACKAGES);
            // Spawn in new process group for clean termination of all children
            c.process_group(0);
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            c
        }
        Language::Javascript => {
            let mut c = Command::new("bun");
            c.arg("add");
            for pkg in packages {
                c.arg(pkg);
            }
            // Install to system path for snapshot persistence (NODE_PATH resolution).
            // Parent of NODE_MODULES_SYSTEM — bun creates node_modules/ here.
            c.current_dir("/usr/local/lib");
            // Spawn in new process group for clean termination of all children
            c.process_group(0);
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            c
        }
        Language::Raw => unreachable!(), // Already validated above
    };

    // Spawn process
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!(
                    "Failed to execute package manager for {}: {}",
                    language.as_str(),
                    e
                ),
                "execution_error",
                None,
            )
            .await?;
            return Ok(());
        }
    };

    // Get stdout/stderr streams
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Spawn independent reader tasks (Tokio best practice Nov 2025 - ZERO race conditions)
    // These tasks continue reading until EOF regardless of when child.wait() completes
    let stdout_task = tokio::spawn(async move {
        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut lines = Vec::new();
        let mut total_bytes = 0usize;

        while let Ok(Some(line)) = stdout_reader.next_line().await {
            if total_bytes + line.len() + 1 > MAX_PACKAGE_OUTPUT_BYTES {
                let remaining = MAX_PACKAGE_OUTPUT_BYTES.saturating_sub(total_bytes);
                if remaining > 0 {
                    lines.push(line[..remaining.min(line.len())].to_string());
                }
                lines.push(format!(
                    "[truncated: output limit {}KB exceeded]",
                    MAX_PACKAGE_OUTPUT_BYTES / 1024
                ));
                break;
            }
            total_bytes += line.len() + 1;
            lines.push(line);
        }
        lines
    });

    let stderr_task = tokio::spawn(async move {
        let mut stderr_reader = BufReader::new(stderr).lines();
        let mut lines = Vec::new();
        let mut total_bytes = 0usize;

        while let Ok(Some(line)) = stderr_reader.next_line().await {
            if total_bytes + line.len() + 1 > MAX_PACKAGE_OUTPUT_BYTES {
                let remaining = MAX_PACKAGE_OUTPUT_BYTES.saturating_sub(total_bytes);
                if remaining > 0 {
                    lines.push(line[..remaining.min(line.len())].to_string());
                }
                lines.push(format!(
                    "[truncated: output limit {}KB exceeded]",
                    MAX_PACKAGE_OUTPUT_BYTES / 1024
                ));
                break;
            }
            total_bytes += line.len() + 1;
            lines.push(line);
        }
        lines
    });

    // Wait for process with timeout
    let wait_result = tokio::time::timeout(
        Duration::from_secs(PACKAGE_INSTALL_TIMEOUT_SECONDS),
        child.wait(),
    )
    .await;

    let status = match wait_result {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            // Graceful termination: SIGTERM → wait → SIGKILL
            let _ = graceful_terminate_process_group(&mut child, TERM_GRACE_PERIOD_SECONDS).await;
            send_streaming_error(
                write_tx,
                format!("Process wait error: {}", e),
                "execution_error",
                None,
            )
            .await?;
            return Ok(());
        }
        Err(_) => {
            // Timeout: Graceful termination: SIGTERM → wait → SIGKILL
            let _ = graceful_terminate_process_group(&mut child, TERM_GRACE_PERIOD_SECONDS).await;
            send_streaming_error(
                write_tx,
                format!(
                    "Package installation timeout after {}s",
                    PACKAGE_INSTALL_TIMEOUT_SECONDS
                ),
                "timeout_error",
                None,
            )
            .await?;
            return Ok(());
        }
    };

    // Reader tasks continue independently - guaranteed to capture ALL output
    let stdout_lines = stdout_task.await.unwrap_or_default();
    let stderr_lines = stderr_task.await.unwrap_or_default();

    let duration_ms = start.elapsed().as_millis() as u64;
    let exit_code = exit_code_from_status(status);

    // Sync filesystem to ensure package files are persisted
    // Critical for snapshots with cache=unsafe (QEMU may exit before lazy writeback)
    // This adds ~5-10ms but guarantees data integrity
    if exit_code == 0 {
        unsafe { libc::sync() };
    }

    // Stream all captured output (batched for efficiency)
    if !stdout_lines.is_empty() {
        send_response(
            write_tx,
            &GuestResponse::Stdout {
                chunk: stdout_lines.join("\n") + "\n",
            },
        )
        .await?;
    }

    if !stderr_lines.is_empty() {
        send_response(
            write_tx,
            &GuestResponse::Stderr {
                chunk: stderr_lines.join("\n") + "\n",
            },
        )
        .await?;
    }

    // Send completion message (no granular timing for install_packages)
    send_response(
        write_tx,
        &GuestResponse::Complete {
            exit_code,
            execution_time_ms: duration_ms,
            spawn_ms: None,
            process_ms: None,
        },
    )
    .await?;

    Ok(())
}

// =============================================================================
// File I/O Handlers
// =============================================================================

/// Validate a relative file path and resolve it under SANDBOX_ROOT.
///
/// Security checks:
/// - Non-empty (except for list_files root)
/// - Length <= MAX_FILE_PATH_LENGTH
/// - No null bytes or control characters
/// - No absolute paths
/// - Path traversal prevention (.. normalization checked against sandbox root)
///
/// For empty path (list_files root), returns SANDBOX_ROOT directly.
fn validate_file_path(relative_path: &str) -> Result<std::path::PathBuf, String> {
    // Empty path = sandbox root (valid for list_files)
    if relative_path.is_empty() {
        return Ok(std::path::PathBuf::from(SANDBOX_ROOT));
    }

    // Length check
    if relative_path.len() > MAX_FILE_PATH_LENGTH {
        return Err(format!(
            "Path too long: {} chars (max {})",
            relative_path.len(),
            MAX_FILE_PATH_LENGTH
        ));
    }

    // Null byte check
    if relative_path.contains('\0') {
        return Err("Path contains null byte".to_string());
    }

    // Control character check
    if relative_path.chars().any(|c| c.is_control()) {
        return Err("Path contains control character".to_string());
    }

    // Absolute path check
    if relative_path.starts_with('/') {
        return Err("Absolute paths are not allowed".to_string());
    }

    // Build resolved path by normalizing components (handle ..)
    let sandbox_root = std::path::PathBuf::from(SANDBOX_ROOT);
    let mut resolved = sandbox_root.clone();

    for component in std::path::Path::new(relative_path).components() {
        match component {
            std::path::Component::Normal(c) => {
                // Per-component byte-length check (NAME_MAX = 255 bytes).
                // ext4/xfs count UTF-8 bytes, not characters, so a filename
                // of 100 CJK chars (300 bytes) would exceed the limit.
                let name_bytes = c.as_encoded_bytes().len();
                if name_bytes > MAX_FILE_NAME_BYTES {
                    return Err(format!(
                        "Filename component too long: {} bytes (max {})",
                        name_bytes, MAX_FILE_NAME_BYTES
                    ));
                }
                resolved.push(c);
            }
            std::path::Component::ParentDir => {
                // Go up one level, but never above sandbox root
                if !resolved.pop() || !resolved.starts_with(&sandbox_root) {
                    return Err("Path traversal outside sandbox".to_string());
                }
            }
            std::path::Component::CurDir => {} // Skip "."
            _ => return Err("Invalid path component".to_string()),
        }
    }

    // Final check: resolved path must be under sandbox root
    if !resolved.starts_with(&sandbox_root) {
        return Err("Path traversal outside sandbox".to_string());
    }

    Ok(resolved)
}

/// Handle read_file command: read from disk, stream as zstd-compressed chunks.
async fn handle_read_file(
    op_id: &str,
    path: &str,
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate path
    let resolved_path = match validate_file_path(path) {
        Ok(p) => p,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Invalid path '{}': {}", path, e),
                "validation_error",
                Some(op_id),
            )
            .await?;
            return Ok(());
        }
    };

    // Reject sandbox root itself
    if resolved_path == std::path::Path::new(SANDBOX_ROOT) {
        send_streaming_error(
            write_tx,
            "Cannot read a directory".to_string(),
            "validation_error",
            Some(op_id),
        )
        .await?;
        return Ok(());
    }

    // Canonicalize to follow symlinks and verify still under sandbox root
    let canonical_path = match tokio::fs::canonicalize(&resolved_path).await {
        Ok(p) => p,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("File not found or inaccessible '{}': {}", path, e),
                "io_error",
                Some(op_id),
            )
            .await?;
            return Ok(());
        }
    };

    let sandbox_canonical = tokio::fs::canonicalize(SANDBOX_ROOT)
        .await
        .unwrap_or_else(|_| std::path::PathBuf::from(SANDBOX_ROOT));
    if !canonical_path.starts_with(&sandbox_canonical) {
        send_streaming_error(
            write_tx,
            format!("Path '{}' resolves outside sandbox", path),
            "validation_error",
            Some(op_id),
        )
        .await?;
        return Ok(());
    }

    // Check it's a regular file
    let metadata = match tokio::fs::metadata(&canonical_path).await {
        Ok(m) => m,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Cannot read '{}': {}", path, e),
                "io_error",
                Some(op_id),
            )
            .await?;
            return Ok(());
        }
    };

    if metadata.is_dir() {
        send_streaming_error(
            write_tx,
            format!("'{}' is a directory, not a file", path),
            "validation_error",
            Some(op_id),
        )
        .await?;
        return Ok(());
    }

    // Check size before reading
    if metadata.len() as usize > MAX_FILE_SIZE_BYTES {
        send_streaming_error(
            write_tx,
            format!(
                "File too large: {} bytes (max {})",
                metadata.len(),
                MAX_FILE_SIZE_BYTES
            ),
            "validation_error",
            Some(op_id),
        )
        .await?;
        return Ok(());
    }

    // Read file content (sync) -> stream as compressed chunks
    let file_size = metadata.len();
    let file = match std::fs::File::open(&canonical_path) {
        Ok(f) => f,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Failed to read '{}': {}", path, e),
                "io_error",
                Some(op_id),
            )
            .await?;
            return Ok(());
        }
    };

    let mut encoder = match zstd::stream::read::Encoder::new(file, FILE_TRANSFER_ZSTD_LEVEL) {
        Ok(e) => e,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Compression init failed for '{}': {}", path, e),
                "io_error",
                Some(op_id),
            )
            .await?;
            return Ok(());
        }
    };

    let mut buf = vec![0u8; FILE_TRANSFER_CHUNK_SIZE];
    loop {
        let n = match std::io::Read::read(&mut encoder, &mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                send_streaming_error(
                    write_tx,
                    format!("Compression error: {}", e),
                    "io_error",
                    Some(op_id),
                )
                .await?;
                return Ok(());
            }
        };
        // TODO: Binary framing would eliminate this base64 encode overhead (~33% wire bloat)
        let chunk_b64 = BASE64.encode(&buf[..n]);
        send_response(
            write_tx,
            &GuestResponse::FileChunk {
                op_id: op_id.to_string(),
                data: chunk_b64,
            },
        )
        .await?;
    }

    // Send completion
    send_response(
        write_tx,
        &GuestResponse::FileReadComplete {
            op_id: op_id.to_string(),
            path: path.to_string(),
            size: file_size,
        },
    )
    .await?;

    Ok(())
}

/// Handle list_files command: read directory entries.
async fn handle_list_files(
    path: &str,
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate path (empty = sandbox root)
    let resolved_path = match validate_file_path(path) {
        Ok(p) => p,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Invalid path '{}': {}", path, e),
                "validation_error",
                None,
            )
            .await?;
            return Ok(());
        }
    };

    // Read directory
    let mut read_dir = match tokio::fs::read_dir(&resolved_path).await {
        Ok(rd) => rd,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Cannot list '{}': {}", path, e),
                "io_error",
                None,
            )
            .await?;
            return Ok(());
        }
    };

    let mut entries = Vec::new();
    while let Ok(Some(entry)) = read_dir.next_entry().await {
        let metadata = match entry.metadata().await {
            Ok(m) => m,
            Err(_) => continue,
        };
        entries.push(FileEntry {
            name: entry.file_name().to_string_lossy().to_string(),
            is_dir: metadata.is_dir(),
            size: if metadata.is_dir() { 0 } else { metadata.len() },
        });
    }

    // Sort entries by name for consistent ordering
    entries.sort_by(|a, b| a.name.cmp(&b.name));

    // Send response
    send_response(
        write_tx,
        &GuestResponse::FileList {
            path: path.to_string(),
            entries,
        },
    )
    .await
}

/// Set up a pipelined file write operation.
///
/// Validates the path, creates parent directories, opens a temp file with
/// zstd decompression, and spawns a blocking task for the I/O pipeline.
/// Returns `Some(ActiveWriteHandle)` on success, or `None` if an error was
/// already sent to the host.
async fn handle_write_file(
    op_id: &str,
    path: &str,
    make_executable: bool,
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<Option<ActiveWriteHandle>, Box<dyn std::error::Error>> {
    // Validate path
    let resolved_path = match validate_file_path(path) {
        Ok(p) => p,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Invalid path '{}': {}", path, e),
                "validation_error",
                Some(op_id),
            )
            .await?;
            return Ok(None);
        }
    };
    // Reject sandbox root itself
    if resolved_path == std::path::Path::new(SANDBOX_ROOT) {
        send_streaming_error(
            write_tx,
            "Cannot write to sandbox root directory".to_string(),
            "validation_error",
            Some(op_id),
        )
        .await?;
        return Ok(None);
    }
    // Create parent directories (sync)
    if let Some(parent) = resolved_path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        send_streaming_error(
            write_tx,
            format!("Failed to create directories: {}", e),
            "io_error",
            Some(op_id),
        )
        .await?;
        return Ok(None);
    }
    // Temp path: <dir>/.wr.<op_id>.tmp (40 chars max).
    // Lives in the same directory for atomic rename (same filesystem).
    // Uses a fixed-length name to avoid NAME_MAX overflow when the
    // target filename is already near 255 bytes.
    let tmp_path = resolved_path
        .parent()
        .unwrap_or(std::path::Path::new(SANDBOX_ROOT))
        .join(format!(".wr.{}.tmp", op_id));
    let file = match std::fs::File::create(&tmp_path) {
        Ok(f) => f,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Failed to create temp file: {}", e),
                "io_error",
                Some(op_id),
            )
            .await?;
            return Ok(None);
        }
    };
    let counting_writer = CountingWriter {
        inner: file,
        count: 0,
        limit: MAX_FILE_SIZE_BYTES,
    };
    let decoder = match zstd::stream::write::Decoder::new(counting_writer) {
        Ok(d) => d,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path);
            send_streaming_error(
                write_tx,
                format!("Decompression init failed: {}", e),
                "io_error",
                Some(op_id),
            )
            .await?;
            return Ok(None);
        }
    };

    // Pipeline: spawn a blocking task for decompression + disk I/O.
    // The main loop sends decoded chunks through the channel while
    // the blocking task decompresses and writes to disk in parallel.
    let (chunk_tx, chunk_rx) = mpsc::channel::<Vec<u8>>(16);
    let write_state = WriteFileState {
        decoder,
        tmp_path,
        final_path: resolved_path,
        make_executable,
        op_id: op_id.to_string(),
        path_display: path.to_string(),
        finished: false,
    };

    let task = tokio::task::spawn_blocking(move || {
        let mut state = write_state;
        let mut chunk_rx = chunk_rx;
        // blocking_recv() is safe here — we're on the blocking thread pool.
        // Empty vec = "finalize" sentinel from FileEnd handler.
        // None (channel closed) without sentinel = abort (error/disconnect).
        let mut finalize = false;
        while let Some(decoded) = chunk_rx.blocking_recv() {
            if decoded.is_empty() {
                finalize = true;
                break;
            }
            if let Err(e) = std::io::Write::write_all(&mut state.decoder, &decoded) {
                // state dropped on return → Drop cleans up tmp file
                return Err(state.io_error(format!("write: {e}")));
            }
        }
        if !finalize {
            // Channel closed without sentinel — abort (sender dropped
            // due to decode error, disconnect, etc.). Don't set
            // finished=true so Drop cleans up the tmp file.
            return Err(state.io_error("channel closed without finalize"));
        }
        // Sentinel received — all chunks delivered, finalize
        if let Err(e) = std::io::Write::flush(&mut state.decoder) {
            return Err(state.io_error(format!("decompression finalize: {e}")));
        }
        let bytes_written = state.decoder.get_ref().count;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = if state.make_executable { 0o755 } else { 0o644 };
            if let Err(e) =
                std::fs::set_permissions(&state.tmp_path, std::fs::Permissions::from_mode(mode))
            {
                return Err(state.io_error(format!("set_permissions: {e}")));
            }
            // Chown to sandbox user so the REPL process can
            // read and modify the file.
            let ret = unsafe {
                libc::chown(
                    std::ffi::CString::new(state.tmp_path.as_os_str().as_encoded_bytes())
                        .unwrap()
                        .as_ptr(),
                    SANDBOX_UID,
                    SANDBOX_GID,
                )
            };
            if ret != 0 {
                let e = std::io::Error::last_os_error();
                return Err(state.io_error(format!("chown: {e}")));
            }
        }
        if let Err(e) = std::fs::rename(&state.tmp_path, &state.final_path) {
            return Err(state.io_error(format!("rename: {e}")));
        }
        state.finished = true;
        Ok(WriteResult {
            bytes_written,
            path_display: state.path_display.clone(),
            op_id: state.op_id.clone(),
        })
    });

    Ok(Some(ActiveWriteHandle {
        chunk_tx,
        task,
        path_display: path.to_string(),
        op_id: op_id.to_string(),
    }))
}

// Validation helper for execute_code_streaming (language already validated via Language::parse)
fn validate_execute_params(
    code: &str,
    timeout: u64,
    env_vars: &HashMap<String, String>,
) -> Result<(), String> {
    // Check code not empty
    if code.trim().is_empty() {
        return Err("Code cannot be empty".to_string());
    }

    // Reject null bytes: shell $() silently truncates at \0, Python compile()
    // rejects them, and JS engine behavior is undefined.
    if code.contains('\0') {
        return Err("Code cannot contain null bytes".to_string());
    }

    // Check code size
    if code.len() > MAX_CODE_SIZE_BYTES {
        return Err(format!(
            "Code too large: {} bytes (max {} bytes)",
            code.len(),
            MAX_CODE_SIZE_BYTES
        ));
    }

    // Check timeout
    if timeout > MAX_TIMEOUT_SECONDS {
        return Err(format!(
            "Timeout too large: {}s (max {}s)",
            timeout, MAX_TIMEOUT_SECONDS
        ));
    }

    // Check environment variables count
    if env_vars.len() > MAX_ENV_VARS {
        return Err(format!(
            "Too many environment variables: {} (max {})",
            env_vars.len(),
            MAX_ENV_VARS
        ));
    }

    // Check each env var
    for (key, value) in env_vars {
        if BLOCKED_ENV_VARS.contains(&key.to_uppercase().as_str()) {
            return Err(format!(
                "Blocked environment variable: '{}' (security risk)",
                key
            ));
        }

        if key.is_empty() || key.len() > MAX_ENV_VAR_NAME_LENGTH {
            return Err(format!(
                "Invalid environment variable name length: {} (max {})",
                key.len(),
                MAX_ENV_VAR_NAME_LENGTH
            ));
        }

        if value.len() > MAX_ENV_VAR_VALUE_LENGTH {
            return Err(format!(
                "Environment variable value too large: {} bytes (max {})",
                value.len(),
                MAX_ENV_VAR_VALUE_LENGTH
            ));
        }

        // Check for control characters in name and value
        // Allows: tab (0x09), printable ASCII (0x20-0x7E), UTF-8 continuation (0x80+)
        // Forbids: NUL, C0 controls (except tab), DEL (0x7F)
        fn is_forbidden_control_char(c: char) -> bool {
            let code = c as u32;
            code < 0x09 || (0x0A..0x20).contains(&code) || code == 0x7F
        }

        if key.chars().any(is_forbidden_control_char) {
            return Err(format!(
                "Environment variable name '{}' contains forbidden control character",
                key
            ));
        }

        if value.chars().any(is_forbidden_control_char) {
            return Err(format!(
                "Environment variable '{}' value contains forbidden control character",
                key
            ));
        }
    }

    Ok(())
}

/// Helper to flush a buffer as a Stdout/Stderr response message
async fn flush_output_buffer(
    write_tx: &mpsc::Sender<Vec<u8>>,
    buffer: &mut String,
    chunk_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if buffer.is_empty() {
        return Ok(());
    }

    let data = std::mem::take(buffer);
    let msg = match chunk_type {
        "stderr" => GuestResponse::Stderr { chunk: data },
        _ => GuestResponse::Stdout { chunk: data },
    };
    send_response(write_tx, &msg).await
}

/// Gracefully terminate a process group: SIGTERM → wait → SIGKILL
///
/// Implements Kubernetes-style graceful shutdown:
/// 1. Send SIGTERM to entire process group (allows cleanup)
/// 2. Wait for grace period
/// 3. If still running, send SIGKILL
///
/// Uses process groups to ensure all child processes are terminated,
/// not just the direct child. This is critical for shell commands
/// that spawn subprocesses.
async fn graceful_terminate_process_group(
    child: &mut tokio::process::Child,
    grace_period_secs: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::time::{Duration, timeout};

    // Get PID (= PGID when process_group(0) was used at spawn)
    let pid = match child.id() {
        Some(id) => id as i32,
        None => {
            // Process already exited - nothing to do
            return Ok(());
        }
    };

    // Phase 1: Send SIGTERM to entire process group
    // Negative PID sends signal to all processes in the group
    // SAFETY: libc::kill is safe with valid signal numbers
    let term_result = unsafe { libc::kill(-pid, libc::SIGTERM) };
    if term_result == -1 {
        let errno = std::io::Error::last_os_error();
        // ESRCH (3) = No such process - already dead, that's fine
        if errno.raw_os_error() != Some(libc::ESRCH) {
            eprintln!("SIGTERM to process group {} failed: {}", pid, errno);
        }
        // Process already dead or error - try to reap
        let _ = child.wait().await;
        return Ok(());
    }

    // Phase 2: Wait for grace period for process to exit gracefully
    match timeout(Duration::from_secs(grace_period_secs), child.wait()).await {
        Ok(Ok(_status)) => {
            // Process exited gracefully within grace period
            return Ok(());
        }
        Ok(Err(e)) => {
            // Wait error - log but continue to SIGKILL
            eprintln!("Wait error after SIGTERM: {}", e);
        }
        Err(_) => {
            // Timeout - process didn't respond to SIGTERM
            eprintln!(
                "Process {} didn't respond to SIGTERM within {}s, sending SIGKILL",
                pid, grace_period_secs
            );
        }
    }

    // Phase 3: Send SIGKILL to entire process group
    let kill_result = unsafe { libc::kill(-pid, libc::SIGKILL) };
    if kill_result == -1 {
        let errno = std::io::Error::last_os_error();
        // ESRCH = already dead, not an error
        if errno.raw_os_error() != Some(libc::ESRCH) {
            eprintln!("SIGKILL to process group {} failed: {}", pid, errno);
        }
    }

    // Reap the process to prevent zombie
    let _ = child.wait().await;

    Ok(())
}

/// Execute code via persistent REPL with streaming output.
///
/// Flow:
/// 1. Get or spawn REPL for this language (persists across calls)
/// 2. Prepend env var setup code, then send via length-prefixed stdin protocol
/// 3. Stream stdout chunks and buffer stderr lines looking for sentinel
/// 4. On sentinel: drain remaining stdout, send completion with exit code
/// 5. On REPL death (EOF): recover exit code via child.wait(), send completion
/// 6. On timeout: kill REPL (respawned on next call), send timeout error
///
/// Sentinel detection uses `find()` (not `starts_with()`) to handle cases where
/// user stderr without trailing newline gets concatenated with the sentinel line.
async fn execute_code_streaming(
    language_str: &str,
    code: &str,
    timeout: u64,
    env_vars: &HashMap<String, String>,
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    use tokio::io::AsyncReadExt;
    use tokio::time::{Duration, interval};

    // Parse language
    let language = match Language::parse(language_str) {
        Some(l) => l,
        None => {
            send_streaming_error(
                write_tx,
                format!(
                    "Unsupported language '{}' (supported: python, javascript, raw)",
                    language_str
                ),
                "validation_error",
                None,
            )
            .await?;
            return Ok(());
        }
    };

    // Validate params (language already validated above)
    if let Err(error_message) = validate_execute_params(code, timeout, env_vars) {
        send_streaming_error(write_tx, error_message, "validation_error", None).await?;
        return Ok(());
    }

    let start = Instant::now();

    // Get or create persistent REPL for this language
    let spawn_start = Instant::now();
    let mut was_fresh_spawn = false;
    let mut repl = {
        let mut states = REPL_STATES.lock().await;
        match states.remove(language.as_str()) {
            Some(mut existing) => {
                // Check if REPL is still alive
                match existing.child.try_wait() {
                    Ok(Some(_)) => {
                        eprintln!("REPL for {} died, spawning fresh", language.as_str());
                        was_fresh_spawn = true;
                        spawn_repl(language).await?
                    }
                    Ok(None) => existing, // Still alive
                    Err(_) => {
                        was_fresh_spawn = true;
                        spawn_repl(language).await?
                    }
                }
            }
            None => {
                was_fresh_spawn = true;
                spawn_repl(language).await?
            }
        }
    };
    let spawn_ms = if was_fresh_spawn {
        Some(spawn_start.elapsed().as_millis() as u64)
    } else {
        Some(0)
    };

    // Generate unique sentinel for this execution
    // Nanosecond timestamp + monotonic counter: unique and long enough
    // to avoid accidental collision with user stderr output
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let count = SENTINEL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let sentinel_id = format!("{nanos}_{count}");
    let sentinel_prefix = format!("__SENTINEL_{}_", sentinel_id);

    // Prepend env var setup to user code
    let full_code = prepend_env_vars(language, code, env_vars);
    let code_bytes = full_code.as_bytes();

    // Write header + code to REPL stdin (header\ncode, then flush)
    let header = format!("{} {}\n", sentinel_id, code_bytes.len());
    let write_result = async {
        repl.stdin.write_all(header.as_bytes()).await?;
        repl.stdin.write_all(code_bytes).await?;
        repl.stdin.flush().await
    }
    .await;
    if let Err(e) = write_result {
        eprintln!("REPL stdin write failed: {}", e);
        let _ = repl.child.kill().await;
        let _ = repl.child.wait().await;
        send_streaming_error(
            write_tx,
            format!("Failed to send code to REPL: {}", e),
            "execution_error",
            None,
        )
        .await?;
        return Ok(());
    }

    let process_start = Instant::now();

    // Stream output until sentinel detected, REPL dies, or timeout
    let mut stdout_buffer = String::new();
    let mut stderr_buffer = String::new();
    let mut stderr_line_buf = String::new();
    let mut stdout_bytes = [0u8; 8192];
    let mut stderr_bytes = [0u8; 8192];
    let mut flush_timer = interval(Duration::from_millis(FLUSH_INTERVAL_MS));
    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut sentinel_exit_code: Option<i32> = None;

    let timeout_duration = if timeout > 0 {
        Duration::from_secs(timeout)
    } else {
        Duration::from_secs(MAX_TIMEOUT_SECONDS)
    };

    let loop_result = tokio::time::timeout(timeout_duration, async {
        loop {
            tokio::select! {
                // Flush timer (50ms) - send buffered data for real-time feel
                _ = flush_timer.tick() => {
                    let _ = flush_output_buffer(write_tx, &mut stdout_buffer, "stdout").await;
                    let _ = flush_output_buffer(write_tx, &mut stderr_buffer, "stderr").await;
                }

                // Read stdout
                result = repl.stdout.read(&mut stdout_bytes), if !stdout_done => {
                    match result {
                        Ok(0) => stdout_done = true,
                        Ok(n) => {
                            stdout_buffer.push_str(&String::from_utf8_lossy(&stdout_bytes[..n]));
                            if stdout_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                let _ = flush_output_buffer(write_tx, &mut stdout_buffer, "stdout").await;
                            }
                        }
                        Err(_) => stdout_done = true,
                    }
                }

                // Read stderr (with sentinel detection)
                result = repl.stderr.read(&mut stderr_bytes), if !stderr_done => {
                    match result {
                        Ok(0) => stderr_done = true,
                        Ok(n) => {
                            let chunk = String::from_utf8_lossy(&stderr_bytes[..n]).into_owned();
                            stderr_line_buf.push_str(&chunk);

                            // Process complete lines looking for sentinel
                            while let Some(nl) = stderr_line_buf.find('\n') {
                                let line = stderr_line_buf[..nl].to_string();
                                stderr_line_buf = stderr_line_buf[nl + 1..].to_string();

                                // Search for sentinel anywhere in line (not just start)
                                // User stderr without trailing \n can be concatenated with sentinel
                                if let Some(sentinel_pos) = line.find(&sentinel_prefix) {
                                    let sentinel_part = &line[sentinel_pos..];
                                    if sentinel_part.ends_with("__") {
                                        // Forward any user stderr before sentinel
                                        if sentinel_pos > 0 {
                                            stderr_buffer.push_str(&line[..sentinel_pos]);
                                        }
                                        // Parse exit code from __SENTINEL_{id}_{code}__
                                        let code_str = &sentinel_part
                                            [sentinel_prefix.len()..sentinel_part.len() - 2];
                                        sentinel_exit_code =
                                            Some(code_str.parse::<i32>().unwrap_or(-1));
                                        continue;
                                    }
                                }
                                // User stderr - buffer for streaming
                                stderr_buffer.push_str(&line);
                                stderr_buffer.push('\n');
                                if stderr_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                    let _ = flush_output_buffer(write_tx, &mut stderr_buffer, "stderr").await;
                                }
                            }
                        }
                        Err(_) => stderr_done = true,
                    }
                }
            }

            // Exit conditions
            if sentinel_exit_code.is_some() {
                // Sentinel detected - drain any remaining stdout (may still be in pipe)
                loop {
                    match tokio::time::timeout(
                        Duration::from_millis(FLUSH_INTERVAL_MS),
                        repl.stdout.read(&mut stdout_bytes),
                    )
                    .await
                    {
                        Ok(Ok(0)) | Err(_) => break, // EOF or timeout - done draining
                        Ok(Ok(n)) => {
                            stdout_buffer
                                .push_str(&String::from_utf8_lossy(&stdout_bytes[..n]));
                        }
                        Ok(Err(_)) => break,
                    }
                }
                break;
            }
            if stdout_done && stderr_done {
                // REPL died (both streams EOF)
                break;
            }
        }
    })
    .await;

    // Flush residual stderr_line_buf (user stderr without trailing newline)
    if !stderr_line_buf.is_empty() {
        stderr_buffer.push_str(&stderr_line_buf);
    }

    // Final flush of any remaining data
    let _ = flush_output_buffer(write_tx, &mut stdout_buffer, "stdout").await;
    let _ = flush_output_buffer(write_tx, &mut stderr_buffer, "stderr").await;

    let duration_ms = start.elapsed().as_millis() as u64;
    let process_ms = Some(process_start.elapsed().as_millis() as u64);

    match loop_result {
        Ok(()) if sentinel_exit_code.is_some() => {
            // Normal completion via sentinel - REPL stays alive for reuse
            let exit_code = sentinel_exit_code.unwrap();
            REPL_STATES
                .lock()
                .await
                .insert(language.as_str().to_string(), repl);

            send_response(
                write_tx,
                &GuestResponse::Complete {
                    exit_code,
                    execution_time_ms: duration_ms,
                    spawn_ms,
                    process_ms,
                },
            )
            .await?;
        }
        Ok(()) => {
            // REPL died (EOF on both streams) - recover exit code
            let status = repl.child.wait().await;
            let exit_code = status.map(exit_code_from_status).unwrap_or(-1);
            eprintln!(
                "REPL for {} died with exit_code={}",
                language.as_str(),
                exit_code
            );

            send_response(
                write_tx,
                &GuestResponse::Complete {
                    exit_code,
                    execution_time_ms: duration_ms,
                    spawn_ms,
                    process_ms,
                },
            )
            .await?;
        }
        Err(_) => {
            // Timeout - kill REPL (will be respawned on next exec)
            eprintln!(
                "REPL for {} timed out after {}s, killing",
                language.as_str(),
                timeout
            );
            let _ =
                graceful_terminate_process_group(&mut repl.child, TERM_GRACE_PERIOD_SECONDS).await;

            send_streaming_error(
                write_tx,
                format!("Execution timeout after {}s", timeout),
                "timeout_error",
                None,
            )
            .await?;
        }
    }

    Ok(())
}

/// Non-blocking file wrapper for virtio-serial ports.
///
/// Uses AsyncFd for true async I/O (epoll-based) instead of tokio::fs::File
/// which uses a blocking threadpool. This enables proper timeout detection:
/// blocking reads can get stuck in kernel space on hung connections, ignoring
/// tokio timeouts. With AsyncFd + epoll, timeouts work correctly and the
/// guest agent can detect and recover from stale connections.
struct NonBlockingFile {
    async_fd: AsyncFd<std::fs::File>,
    leftover: Vec<u8>, // bytes read past the last newline
}

impl NonBlockingFile {
    /// Open a file with O_NONBLOCK and wrap it for async I/O.
    fn open_read(path: &str) -> std::io::Result<Self> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new().read(true).open(path)?;
        Self::set_nonblocking(file.as_raw_fd())?;
        let async_fd = AsyncFd::new(file)?;
        Ok(Self {
            async_fd,
            leftover: Vec::new(),
        })
    }

    /// Set O_NONBLOCK on a file descriptor using fcntl.
    fn set_nonblocking(fd: RawFd) -> std::io::Result<()> {
        // Get current flags
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 {
            return Err(std::io::Error::last_os_error());
        }

        // Set O_NONBLOCK
        let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
        if result < 0 {
            return Err(std::io::Error::last_os_error());
        }

        Ok(())
    }

    /// Check if the host side of the virtio-serial port is connected.
    ///
    /// Uses poll() to detect EPOLLHUP which the kernel's virtio_console driver
    /// sets when `port->host_connected` is false. This allows the agent to
    /// detect disconnection without busy-looping on read attempts.
    ///
    /// Returns:
    /// - Ok(true) if host is connected (no POLLHUP)
    /// - Ok(false) if host is disconnected (POLLHUP set)
    /// - Err on poll failure
    fn is_host_connected(&self) -> std::io::Result<bool> {
        let fd = self.async_fd.get_ref().as_raw_fd();

        let mut pollfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };

        // Poll with 0 timeout (non-blocking check)
        let result = unsafe { libc::poll(&mut pollfd, 1, 0) };

        if result < 0 {
            return Err(std::io::Error::last_os_error());
        }

        // POLLHUP indicates host disconnected (virtio_console sets this when !host_connected)
        let host_disconnected = (pollfd.revents & libc::POLLHUP) != 0;
        Ok(!host_disconnected)
    }

    /// Read a line with proper async timeout support and buffered I/O.
    ///
    /// Unlike tokio::fs::File::read which uses spawn_blocking (and thus
    /// can't be interrupted by timeout when stuck in kernel), this method
    /// uses AsyncFd::readable() which properly integrates with tokio's
    /// event loop and can be cancelled by timeout.
    ///
    /// Reads up to 16KB per syscall instead of 1 byte, reducing syscalls
    /// from ~175,000 to ~11 per chunk message (~16,000x reduction).
    /// Bytes past the newline are stored in `leftover` for the next call.
    async fn read_line(&mut self, buf: &mut String) -> std::io::Result<usize> {
        let mut total_bytes = 0;
        // Accumulate raw bytes for proper UTF-8 decoding
        // (pushing bytes directly as chars corrupts multi-byte UTF-8 sequences)
        let mut bytes = Vec::new();

        // First drain leftover from previous read
        if !self.leftover.is_empty() {
            if let Some(pos) = self.leftover.iter().position(|&b| b == b'\n') {
                // Found newline in leftover — return up to and including it
                let line_len = pos + 1;
                buf.push_str(&String::from_utf8_lossy(&self.leftover[..line_len]));
                self.leftover.drain(..line_len);
                return Ok(line_len);
            }
            // No newline in leftover — consume all of it and keep reading
            total_bytes += self.leftover.len();
            bytes.append(&mut self.leftover);
        }

        let mut read_buf = [0u8; 16384]; // 16KB read buffer

        loop {
            // Wait for the fd to be readable (epoll-based, properly cancellable)
            let mut guard = self.async_fd.readable().await?;

            // Try to read up to 16KB (non-blocking)
            match guard.try_io(|inner| {
                let fd = inner.get_ref().as_raw_fd();
                let result = unsafe {
                    libc::read(
                        fd,
                        read_buf.as_mut_ptr() as *mut libc::c_void,
                        read_buf.len(),
                    )
                };
                if result < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(result as usize)
                }
            }) {
                Ok(Ok(0)) => {
                    // EOF - convert accumulated bytes to UTF-8
                    buf.push_str(&String::from_utf8_lossy(&bytes));
                    return Ok(total_bytes);
                }
                Ok(Ok(n)) => {
                    let chunk = &read_buf[..n];
                    if let Some(pos) = chunk.iter().position(|&b| b == b'\n') {
                        // Found newline — take up to and including it
                        bytes.extend_from_slice(&chunk[..=pos]);
                        total_bytes += pos + 1;
                        // Store remaining bytes after newline for next call
                        if pos + 1 < n {
                            self.leftover.extend_from_slice(&chunk[pos + 1..]);
                        }
                        buf.push_str(&String::from_utf8_lossy(&bytes));
                        return Ok(total_bytes);
                    }
                    // No newline — accumulate and keep reading
                    total_bytes += n;
                    bytes.extend_from_slice(chunk);
                }
                Ok(Err(e)) => {
                    return Err(e);
                }
                Err(_would_block) => {
                    // Spurious wakeup, continue waiting
                    continue;
                }
            }
        }
    }
}

async fn run_with_ports(
    mut cmd_file: NonBlockingFile,
    event_file: tokio::fs::File,
) -> Result<(), Box<dyn std::error::Error>> {
    // Use the non-blocking file directly for reads
    // Event file can stay as tokio::fs::File since writes don't block

    // Run the connection handler with non-blocking cmd reader
    handle_connection_nonblocking(&mut cmd_file, event_file).await
}

/// Handle connection with non-blocking command reader.
///
/// Uses NonBlockingFile for the command port, enabling proper timeout
/// detection on hung/stale connections.
async fn handle_connection_nonblocking(
    cmd_reader: &mut NonBlockingFile,
    event_file: tokio::fs::File,
) -> Result<(), Box<dyn std::error::Error>> {
    let writer = BufWriter::new(event_file);

    // Create bounded channel for write queue to prevent deadlocks
    let (write_tx, mut write_rx) = mpsc::channel::<Vec<u8>>(WRITE_QUEUE_SIZE);

    // Spawn write task — batches up to 16 writes before flushing to reduce
    // syscalls. Flushes immediately when the channel is empty to keep
    // latency-sensitive single messages (pong, ack) responsive.
    let write_handle = tokio::spawn(async move {
        let mut writer = writer;
        let mut pending = 0u32;
        while let Some(data) = write_rx.recv().await {
            if let Err(e) = writer.write_all(&data).await {
                eprintln!("Write error: {}", e);
                break;
            }
            pending += 1;
            if pending >= 16 || write_rx.is_empty() {
                if let Err(e) = writer.flush().await {
                    eprintln!("Flush error: {}", e);
                    break;
                }
                pending = 0;
            }
        }
        // Final flush for any remaining data
        let _ = writer.flush().await;
    });

    // Active streaming file write operations (pipelined via spawn_blocking)
    let mut active_writes: HashMap<String, ActiveWriteHandle> = HashMap::new();

    // Main loop: read requests, queue responses
    let mut line = String::new();
    let result = loop {
        line.clear();

        // Read request with timeout using non-blocking I/O
        // This timeout will actually work because AsyncFd::readable() is properly
        // cancellable, unlike tokio::fs::File which uses blocking threadpool
        let read_result = tokio::time::timeout(
            std::time::Duration::from_millis(READ_TIMEOUT_MS),
            cmd_reader.read_line(&mut line),
        )
        .await;

        let bytes_read = match read_result {
            Ok(Ok(0)) => {
                eprintln!("Connection closed by client");
                break Ok(());
            }
            Ok(Ok(n)) => n,
            Ok(Err(e)) => {
                eprintln!("Read error: {}", e);
                break Err(e.into());
            }
            Err(_) => {
                // Timeout - hung or stale connection
                // AsyncFd enables proper timeout (unlike blocking I/O which ignores it)
                eprintln!("Read timeout after {}ms, reconnecting...", READ_TIMEOUT_MS);
                break Err("read timeout - triggering reconnect".into());
            }
        };

        // Validate request size
        if bytes_read > MAX_REQUEST_SIZE_BYTES {
            let _ = send_streaming_error(
                &write_tx,
                format!(
                    "Request too large: {} bytes (max {} bytes)",
                    bytes_read, MAX_REQUEST_SIZE_BYTES
                ),
                "request_error",
                None,
            )
            .await;
            continue;
        }

        // Log request for debugging
        eprintln!("Received request ({} bytes)", bytes_read);

        // Parse and execute command
        match serde_json::from_str::<GuestCommand>(&line) {
            Ok(GuestCommand::Ping) => {
                eprintln!("Processing: ping");
                if send_response(&write_tx, &GuestResponse::Pong { version: VERSION })
                    .await
                    .is_err()
                {
                    break Err("write queue closed".into());
                }
            }
            Ok(GuestCommand::WarmRepl { language }) => {
                eprintln!("Processing: warm_repl (language={})", language);
                let lang = match Language::parse(&language) {
                    Some(l) => l,
                    None => {
                        if send_response(
                            &write_tx,
                            &GuestResponse::WarmReplAck {
                                language,
                                status: "error".to_string(),
                                message: Some(
                                    "Unsupported language (supported: python, javascript, raw)"
                                        .to_string(),
                                ),
                            },
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                        continue;
                    }
                };
                match spawn_repl(lang).await {
                    Ok(repl) => {
                        REPL_STATES.lock().await.insert(language.clone(), repl);
                        eprintln!("REPL pre-warmed for {}", language);
                        if send_response(
                            &write_tx,
                            &GuestResponse::WarmReplAck {
                                language,
                                status: "ok".to_string(),
                                message: None,
                            },
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: eager REPL spawn failed: {}", e);
                        if send_response(
                            &write_tx,
                            &GuestResponse::WarmReplAck {
                                language,
                                status: "error".to_string(),
                                message: Some(e.to_string()),
                            },
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                }
            }
            Ok(GuestCommand::InstallPackages { language, packages }) => {
                eprintln!(
                    "Processing: install_packages (language={}, count={})",
                    language,
                    packages.len()
                );
                let lang = match Language::parse(&language) {
                    Some(l) => l,
                    None => {
                        if send_streaming_error(
                            &write_tx,
                            format!(
                                "Unsupported language '{}' for package installation (supported: python, javascript)",
                                language
                            ),
                            "validation_error",
                            None,
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                        continue;
                    }
                };
                if install_packages(lang, &packages, &write_tx).await.is_err() {
                    break Err("install_packages failed".into());
                }
            }
            Ok(GuestCommand::ExecuteCode {
                language,
                code,
                timeout,
                env_vars,
            }) => {
                eprintln!(
                    "Processing: execute_code (language={}, code_size={}, timeout={}s, env_vars={})",
                    language,
                    code.len(),
                    timeout,
                    env_vars.len()
                );
                if execute_code_streaming(&language, &code, timeout, &env_vars, &write_tx)
                    .await
                    .is_err()
                {
                    break Err("execute_code failed".into());
                }
            }
            Ok(GuestCommand::WriteFile {
                op_id,
                path,
                make_executable,
            }) => {
                eprintln!(
                    "Processing: write_file (op_id={}, path={}, executable={})",
                    op_id, path, make_executable
                );
                match handle_write_file(&op_id, &path, make_executable, &write_tx).await {
                    Ok(Some(handle)) => {
                        active_writes.insert(op_id, handle);
                    }
                    Ok(None) => {} // error already sent to host
                    Err(e) => {
                        break Err(e);
                    }
                }
            }
            Ok(GuestCommand::FileChunk { op_id, data }) => {
                if !active_writes.contains_key(&op_id) {
                    let _ = send_streaming_error(
                        &write_tx,
                        format!("No active write for op_id '{}'", op_id),
                        "protocol_error",
                        Some(&op_id),
                    )
                    .await;
                    continue;
                }
                // TODO: Binary framing would eliminate this base64 decode overhead (~33% wire bloat)
                let decoded = match BASE64.decode(&data) {
                    Ok(d) => d,
                    Err(e) => {
                        // Remove handle and drop chunk_tx — the blocking task
                        // will see channel close without sentinel and abort,
                        // cleaning up the tmp file via Drop.
                        let handle = active_writes.remove(&op_id).unwrap();
                        drop(handle.chunk_tx);
                        let _ = send_streaming_error(
                            &write_tx,
                            format!(
                                "Invalid base64 in chunk for '{}': {}",
                                handle.path_display, e
                            ),
                            "validation_error",
                            Some(&op_id),
                        )
                        .await;
                        continue;
                    }
                };
                // Non-blocking send to the pipeline worker. If the channel is full
                // (16 items), this awaits briefly, providing natural backpressure.
                // We use get() here — the borrow is released before any mutation.
                let send_failed = {
                    let handle = active_writes.get(&op_id).unwrap();
                    handle.chunk_tx.send(decoded).await.is_err()
                };
                if send_failed {
                    // Worker task exited with an error — remove handle and
                    // await the task to surface the actual error message.
                    let handle = active_writes.remove(&op_id).unwrap();
                    drop(handle.chunk_tx);
                    match handle.task.await {
                        Ok(Err(write_err)) => {
                            eprintln!(
                                "Write pipeline error for '{}' (op_id={}): {}",
                                write_err.path_display, write_err.op_id, write_err.message
                            );
                            let _ = send_streaming_error(
                                &write_tx,
                                write_err.message,
                                &write_err.error_type,
                                Some(&write_err.op_id),
                            )
                            .await;
                        }
                        Ok(Ok(_)) => {
                            // Task succeeded but channel closed — shouldn't happen
                            let _ = send_streaming_error(
                                &write_tx,
                                format!(
                                    "Write pipeline unexpectedly closed for '{}'",
                                    handle.path_display
                                ),
                                "io_error",
                                Some(&op_id),
                            )
                            .await;
                        }
                        Err(join_err) => {
                            let _ = send_streaming_error(
                                &write_tx,
                                format!(
                                    "Internal error writing '{}': {}",
                                    handle.path_display, join_err
                                ),
                                "io_error",
                                Some(&handle.op_id),
                            )
                            .await;
                        }
                    }
                    continue;
                }
            }
            Ok(GuestCommand::FileEnd { op_id }) => {
                let ActiveWriteHandle {
                    chunk_tx,
                    task,
                    path_display,
                    op_id: handle_op_id,
                } = match active_writes.remove(&op_id) {
                    Some(h) => h,
                    None => {
                        let _ = send_streaming_error(
                            &write_tx,
                            format!("No active write for op_id '{}'", op_id),
                            "protocol_error",
                            Some(&op_id),
                        )
                        .await;
                        continue;
                    }
                };
                // Send empty vec as "finalize" sentinel, then drop sender.
                // The blocking task distinguishes this from channel-close-on-error.
                let _ = chunk_tx.send(vec![]).await;
                drop(chunk_tx);
                // Await the blocking task result
                match task.await {
                    Ok(Ok(result)) => {
                        if send_response(
                            &write_tx,
                            &GuestResponse::FileWriteAck {
                                op_id: result.op_id,
                                path: result.path_display,
                                bytes_written: result.bytes_written,
                            },
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                    Ok(Err(write_err)) => {
                        eprintln!(
                            "Write pipeline error for '{}' (op_id={}): {}",
                            write_err.path_display, write_err.op_id, write_err.message
                        );
                        let _ = send_streaming_error(
                            &write_tx,
                            write_err.message,
                            &write_err.error_type,
                            Some(&write_err.op_id),
                        )
                        .await;
                    }
                    Err(join_err) => {
                        let _ = send_streaming_error(
                            &write_tx,
                            format!("Internal error writing '{}': {}", path_display, join_err),
                            "io_error",
                            Some(&handle_op_id),
                        )
                        .await;
                    }
                }
            }
            Ok(GuestCommand::ReadFile { op_id, path }) => {
                eprintln!("Processing: read_file (op_id={}, path={})", op_id, path);
                if handle_read_file(&op_id, &path, &write_tx).await.is_err() {
                    break Err("read_file failed".into());
                }
            }
            Ok(GuestCommand::ListFiles { path }) => {
                eprintln!("Processing: list_files (path={})", path);
                if handle_list_files(&path, &write_tx).await.is_err() {
                    break Err("list_files failed".into());
                }
            }
            Err(e) => {
                eprintln!("JSON parse error: {}", e);
                let _ = send_streaming_error(
                    &write_tx,
                    format!("Invalid JSON: {}", e),
                    "request_error",
                    None,
                )
                .await;
            }
        }
    };

    // Drop write_tx to signal write task to exit
    drop(write_tx);

    // Wait for write task to finish
    let _ = write_handle.await;

    result
}

async fn listen_virtio_serial() -> Result<(), Box<dyn std::error::Error>> {
    use tokio::fs::OpenOptions;

    // Open dual virtio-serial ports (created by QEMU virtserialport)
    // CMD port: host → guest (read-only) - uses NonBlockingFile for timeout support
    // EVENT port: guest → host (write-only) - uses tokio::fs::File
    eprintln!(
        "Guest agent opening virtio-serial ports: cmd={}, event={}",
        CMD_PORT_PATH, EVENT_PORT_PATH
    );

    // Exponential backoff state for host disconnection
    // This prevents busy-looping when host is not connected (EPOLLHUP case)
    let mut backoff_ms = INITIAL_BACKOFF_MS;

    loop {
        // Open command port with O_NONBLOCK for proper timeout support
        // Enables detection of hung/stale connections
        let cmd_file = match NonBlockingFile::open_read(CMD_PORT_PATH) {
            Ok(f) => {
                eprintln!("Guest agent connected to command port (read, non-blocking)");
                f
            }
            Err(e) => {
                eprintln!(
                    "Failed to open command port: {}, retrying in {}ms...",
                    e, backoff_ms
                );
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
        };

        // Check if host is actually connected before proceeding
        // The kernel's virtio_console driver sets POLLHUP when host_connected=false.
        // Without this check, we'd busy-loop on read() returning EOF immediately.
        match cmd_file.is_host_connected() {
            Ok(true) => {
                eprintln!("Host is connected, proceeding with connection setup");
                // Reset backoff on successful connection
                backoff_ms = INITIAL_BACKOFF_MS;
            }
            Ok(false) => {
                eprintln!(
                    "Host not connected (POLLHUP), waiting {}ms before retry...",
                    backoff_ms
                );
                // Drop the file before sleeping to release kernel resources
                drop(cmd_file);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
            Err(e) => {
                eprintln!(
                    "Failed to check host connection status: {}, retrying in {}ms...",
                    e, backoff_ms
                );
                drop(cmd_file);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
        }

        // Open event port (guest → host, write-only)
        // Write port can use tokio::fs::File - writes don't block in the same way
        let event_file = match OpenOptions::new().write(true).open(EVENT_PORT_PATH).await {
            Ok(f) => {
                eprintln!("Guest agent connected to event port (write)");
                f
            }
            Err(e) => {
                eprintln!(
                    "Failed to open event port: {}, retrying in {}ms...",
                    e, backoff_ms
                );
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                continue;
            }
        };

        // Handle connection with non-blocking cmd reader
        // Note: We pass ownership of files to run_with_ports which ensures
        // they are dropped when it returns (before we try to reopen)
        if let Err(e) = run_with_ports(cmd_file, event_file).await {
            eprintln!("Connection error: {}, reopening ports...", e);
            // Small delay before reconnecting to give kernel time to release
            tokio::time::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS)).await;
            // Don't reset backoff here - if host disconnected, we want to back off
        }
    }
}

/// Reap zombie processes when running as PID 1.
///
/// PID 1 is responsible for reaping orphaned child processes.
/// This async task listens for SIGCHLD signals and calls waitpid()
/// to clean up zombie processes.
///
/// Reference: https://github.com/fpco/pid1-rs
async fn reap_zombies() {
    use tokio::signal::unix::{SignalKind, signal};

    // Create signal stream BEFORE any children are spawned to avoid race conditions
    let mut sigchld = match signal(SignalKind::child()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: Failed to register SIGCHLD handler: {}", e);
            return;
        }
    };

    loop {
        // Wait for SIGCHLD (cancel-safe)
        sigchld.recv().await;

        // Reap all available zombies in a non-blocking loop
        // Multiple children may have exited before we get here
        loop {
            // SAFETY: waitpid with WNOHANG is safe and returns immediately
            let pid = unsafe { libc::waitpid(-1, std::ptr::null_mut(), libc::WNOHANG) };
            match pid {
                p if p > 0 => continue, // Reaped one zombie, check for more
                0 => break,             // No more zombies waiting
                _ => break,             // Error (ECHILD = no children)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Init-time hardening helpers (PID 1 only)
// ---------------------------------------------------------------------------

/// Set file mode on multiple paths (non-fatal, logs nothing on failure).
fn chmod_paths(mode: &str, paths: &[&str]) {
    for path in paths {
        let _ = StdCommand::new("chmod").args([mode, path]).status();
    }
}

/// Bind-mount a path onto itself and remount read-only with nosuid.
///
/// Two-step process required by Linux VFS:
///   1. `mount --rbind <path> <path>` — recursive bind preserving child mounts
///   2. `mount -o remount,ro,nosuid,bind <path>` — makes top mount read-only
///
/// The recursive bind (MS_REC) ensures child mounts (e.g., /dev/shm inside /dev)
/// remain visible and writable. The non-recursive remount only applies read-only
/// to the top mount, not to children.
///
/// Note: MS_NODEV is intentionally excluded — this function is used for /dev
/// (devtmpfs), where MS_NODEV would block all device file access including
/// virtio-serial ports needed for guest-agent communication.
///
/// Returns `true` on success. Logs warnings on failure (non-fatal).
fn mount_readonly(path: &std::ffi::CStr) -> bool {
    unsafe {
        // Step 1: recursive bind mount onto itself.
        // MS_REC preserves child mounts (e.g., /dev/shm tmpfs inside /dev).
        // Without MS_REC, bind-mounting /dev hides /dev/shm, breaking
        // POSIX semaphores (multiprocessing, etc.).
        let ret = libc::mount(
            path.as_ptr(),
            path.as_ptr(),
            std::ptr::null(),
            libc::MS_BIND | libc::MS_REC,
            std::ptr::null(),
        );
        if ret != 0 {
            eprintln!(
                "Warning: bind mount {} failed: {}",
                path.to_string_lossy(),
                std::io::Error::last_os_error()
            );
            return false;
        }
        // Step 2: remount read-only with nosuid
        let ret = libc::mount(
            std::ptr::null(),
            path.as_ptr(),
            std::ptr::null(),
            libc::MS_BIND | libc::MS_REMOUNT | libc::MS_RDONLY | libc::MS_NOSUID,
            std::ptr::null(),
        );
        if ret != 0 {
            eprintln!(
                "Warning: RO remount {} failed: {}",
                path.to_string_lossy(),
                std::io::Error::last_os_error()
            );
            return false;
        }
        true
    }
}

/// Handles userspace-only initialization after tiny-init:
/// - tiny-init: kernel modules, zram, devtmpfs, read-only rootfs mount (initramfs)
/// - guest-agent: PATH, env vars, /home/user tmpfs, network IP config (userspace only)
///
/// Note: modprobe/insmod won't work here - kernel modules are only in initramfs
/// which is unmounted after switch_root.
fn setup_init_environment() {
    // Set PATH for child processes (uv, python3, bun, etc.)
    // SAFETY: called at startup before any threads are spawned
    unsafe { std::env::set_var("PATH", "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin") };
    eprintln!("Set PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin");

    // Disable uv cache (ephemeral VMs)
    // SAFETY: called at startup before any threads are spawned
    unsafe { std::env::set_var("UV_NO_CACHE", "1") };
    eprintln!("Set UV_NO_CACHE=1");

    // Harden system directories: CIS Benchmark 6.1.x compliance.
    // No-op on read-only rootfs (build-time permissions are authoritative).
    // Retained as defense-in-depth against the macOS Docker VirtioFS ownership
    // bug where export loses UID 0.
    // See: https://github.com/docker/for-mac/issues/6812
    chmod_paths("755", &["/etc", "/usr", "/var", "/sbin", "/bin"]);
    // CIS 6.1.2-6.1.4: /etc/passwd, /etc/group = 644; /etc/shadow = 640
    chmod_paths(
        "644",
        &[
            "/etc/passwd",
            "/etc/group",
            "/etc/resolv.conf",
            "/etc/hosts",
        ],
    );
    chmod_paths("640", &["/etc/shadow"]);
    eprintln!("Hardened system directory permissions (CIS 6.1.x)");

    // Mount tmpfs on /home/user — writable scratch space on read-only rootfs.
    // /home/user on rootfs is empty (cloudpickle moved to PYTHON_SITE_PACKAGES).
    //
    // Why tmpfs over alternatives:
    //   - ramfs: no size cap, so runaway writes OOM the VM silently (no ENOSPC back-pressure)
    //   - virtio-blk: virtio exit overhead per I/O, requires host-side image create/attach/cleanup,
    //     and data persists on host disk (undesirable for ephemeral sandboxes)
    //   - virtiofs: requires virtiofsd daemon per VM, adds host attack surface; works on microvm
    //     via vhost-user-fs-device (MMIO) but unnecessary for scratch that doesn't need host access
    //   - overlayfs: /home/user is empty on rootfs, overlay over empty dir = just tmpfs with extra
    //     indirection (and overlayfs requires a writable upperdir, which is itself tmpfs)
    //
    // Memory accounting: pages allocated on demand from guest RAM (not host disk).
    // Default size = 50% of guest RAM; counts toward the QEMU process cgroup memory limit on host.
    //
    // No explicit size — defaults to half-of-RAM, auto-adapts to memory_mb.
    // No noexec — uv/bun install executables into site-packages/node_modules.
    // No nr_inodes cap — package managers create thousands of files.
    // noswap prevents data leaking to zram swap (kernel 6.4+).
    let home_ret = unsafe {
        let source = std::ffi::CString::new("tmpfs").unwrap();
        let target = std::ffi::CString::new("/home/user").unwrap();
        let fstype = std::ffi::CString::new("tmpfs").unwrap();
        let data = std::ffi::CString::new(format!(
            "mode=0755,uid={SANDBOX_UID},gid={SANDBOX_GID},noswap"
        ))
        .unwrap();
        libc::mount(
            source.as_ptr(),
            target.as_ptr(),
            fstype.as_ptr(),
            libc::MS_NOSUID | libc::MS_NODEV,
            data.as_ptr() as *const libc::c_void,
        )
    };
    if home_ret == 0 {
        eprintln!("Mounted tmpfs on /home/user (nosuid,nodev)");
    } else {
        eprintln!(
            "Warning: tmpfs mount on /home/user failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // Read-only bind remounts of /usr, /bin, /sbin (CIS hardening).
    // Skipped when init.rw=1 is set (snapshot creation): package install writes
    // to PYTHON_SITE_PACKAGES and NODE_MODULES_SYSTEM.
    let rw_mode = std::fs::read_to_string("/proc/cmdline")
        .unwrap_or_default()
        .split_ascii_whitespace()
        .any(|p| p == "init.rw=1");

    if rw_mode {
        eprintln!("init.rw=1: skipping /usr, /bin, /sbin read-only remounts");
    } else {
        // Read-only bind remount of /usr with nosuid (CIS: nodev,ro for /usr).
        // Defense-in-depth: rootfs is already ro from tiny-init, this adds a
        // nosuid bind-mount layer on /usr specifically.
        // Does NOT include noexec — system binaries in /usr/bin must remain executable.
        if mount_readonly(c"/usr") {
            eprintln!("Mounted /usr read-only with nosuid");
        }

        // Read-only bind remount of /bin with nosuid (defense-in-depth).
        // Alpine 3.23+: /bin is a symlink to /usr/bin (usrmerge); mount(2) resolves
        // symlinks, so this bind-mounts the already-RO /usr/bin (harmless no-op).
        // Primary protection is the RO rootfs from tiny-init; this adds nosuid layer.
        if mount_readonly(c"/bin") {
            eprintln!("Mounted /bin read-only with nosuid");
        }

        // Read-only bind remount of /sbin with nosuid (defense-in-depth).
        // Alpine /sbin has networking (ip, iptables), filesystem (mkfs, fdisk),
        // and service tools (rc-service). Same usrmerge compatibility as /bin.
        if mount_readonly(c"/sbin") {
            eprintln!("Mounted /sbin read-only with nosuid");
        }
    }

    // Read-only bind remount of /etc/hosts (prevents host alias injection).
    // musl checks /etc/hosts before DNS — injection would affect all name
    // resolution. Defense-in-depth: rootfs is already ro from tiny-init,
    // this adds a nosuid bind-mount layer on /etc/hosts specifically.
    if mount_readonly(c"/etc/hosts") {
        eprintln!("Mounted /etc/hosts read-only");
    }

    // Read-only bind remount of /etc/resolv.conf (prevents DNS hijack).
    // Guest DNS must use gvproxy gateway (192.168.127.1); modifying resolv.conf
    // to point at external DNS (e.g. 8.8.8.8) would bypass filtering intent.
    // Defense-in-depth: rootfs is already ro from tiny-init,
    // this adds a nosuid bind-mount layer on /etc/resolv.conf specifically.
    if mount_readonly(c"/etc/resolv.conf") {
        eprintln!("Mounted /etc/resolv.conf read-only");
    }

    // Read-only bind remount of /dev (defense-in-depth against device node creation).
    // Prevents mknod(2) with EROFS even if CAP_MKNOD were somehow available.
    // Does NOT break device I/O: VFS skips write-access counting for special files
    // (char/block devices go through device driver, not filesystem).
    // Does NOT affect /dev/shm: recursive bind (MS_REC) preserves the child tmpfs
    // mount, and the non-recursive remount-ro only applies to the top devtmpfs.
    // See: CVE-2020-2023 (Kata mknod+debugfs attack chain)
    if mount_readonly(c"/dev") {
        eprintln!("Mounted /dev read-only with nosuid");
    }

    // Read-only bind remount of /proc/sys (prevents sysctl modification).
    // Blocks the most dangerous attack vectors:
    // - core_pattern pipe-to-binary (arbitrary root code execution on crash)
    // - modprobe path hijack (root code execution on unknown binary format)
    // - randomize_va_space=0 (disables ASLR system-wide)
    // - ip_forward (network pivoting), hostname, panic, panic_on_oom (DoS)
    // All sysctl tuning (page-cluster, swappiness, etc.) is done by tiny-init
    // during setup_zram() before switch_root, so no writes needed here.
    // See: CVE-2025-31133, CVE-2025-52565, CVE-2025-52881 (runc /proc/sys escapes)
    if mount_readonly(c"/proc/sys") {
        eprintln!("Mounted /proc/sys read-only");
    }

    // Read-only bind remount of /proc/sysrq-trigger (DoS prevention).
    // /proc/sysrq-trigger is outside /proc/sys (not covered by the remount above).
    // Writing 'c' = kernel crash, 'b' = reboot — denial of service against the VM.
    // IMPORTANT: kernel.sysrq=0 does NOT protect this file — the kernel's
    // write_sysrq_trigger() in drivers/tty/sysrq.c passes check_mask=false to
    // __handle_sysrq(), bypassing the sysrq_enabled bitmask entirely. Only
    // filesystem-level blocking (EROFS via bind-mount) is effective.
    // File permissions (S_IWUSR) block the sandbox user, but this also blocks root.
    // See: CVE-2025-31133, CVE-2025-52565, CVE-2025-52881 (runc procfs redirects)
    if mount_readonly(c"/proc/sysrq-trigger") {
        eprintln!("Mounted /proc/sysrq-trigger read-only");
    }

    // Bring up loopback interface (required for localhost/127.0.0.1 connectivity).
    // In microvm guests with custom init, lo starts DOWN — no systemd/OpenRC to activate it.
    // Without lo, user code like http.createServer + fetch('http://localhost:...') fails
    // with Bun's "FailedToOpenSocket: Was there a typo in the url or port?" error.
    let _ = StdCommand::new("ip")
        .args(["link", "set", "lo", "up"])
        .status();
    eprintln!("Loopback interface up");

    // Wait for network interface (up to 1 second, 20ms intervals)
    // virtio_net loaded by tiny-init, eth0 appears shortly after
    for _ in 0..50 {
        if std::path::Path::new("/sys/class/net/eth0").exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // Configure network for gvproxy mode
    // gvproxy gateway at 192.168.127.1 provides DNS
    // Static IP (DHCP unavailable - AF_PACKET not supported)
    if std::path::Path::new("/sys/class/net/eth0").exists() {
        eprintln!("Configuring network...");

        let _ = StdCommand::new("ip")
            .args(["link", "set", "eth0", "up"])
            .status();
        let _ = StdCommand::new("ip")
            .args(["addr", "add", "192.168.127.2/24", "dev", "eth0"])
            .status();
        let _ = StdCommand::new("ip")
            .args(["route", "add", "default", "via", "192.168.127.1"])
            .status();

        // Note: /etc/resolv.conf is pre-configured in qcow2 image (see build-qcow2.sh)
        eprintln!("Network configured: 192.168.127.2/24 via 192.168.127.1");

        // Verify gvproxy connectivity (exponential backoff: 50+100+200+400+800+1000+1000 = 3550ms max)
        // Ping the gateway to ensure gvproxy is reachable before package install
        let mut gvproxy_ok = false;
        for delay_ms in [50, 100, 200, 400, 800, 1000, 1000] {
            let result = StdCommand::new("ping")
                .args(["-c", "1", "-W", "1", "192.168.127.1"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            if let Ok(status) = result
                && status.success()
            {
                eprintln!("gvproxy connectivity verified");
                gvproxy_ok = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }
        if !gvproxy_ok {
            eprintln!("Warning: gvproxy not reachable, package install may fail");
        }
    } else {
        eprintln!("Warning: eth0 not found, network unavailable");
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // When running as PID 1, setup environment and zombie reaping
    // This must be done early before any child processes are spawned
    if std::process::id() == 1 {
        eprintln!("Guest agent running as PID 1 (init)...");

        // Setup environment (PATH, UV_NO_CACHE, network)
        setup_init_environment();

        // Enable zombie reaper
        eprintln!("Enabling zombie reaper...");
        tokio::spawn(reap_zombies());
    }

    eprintln!(
        "Guest agent starting (dual ports: cmd={}, event={})...",
        CMD_PORT_PATH, EVENT_PORT_PATH
    );
    listen_virtio_serial().await
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // validate_file_path tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_file_path_basic() {
        let result = validate_file_path("hello.txt");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/home/user/hello.txt")
        );
    }

    #[test]
    fn test_validate_file_path_nested() {
        let result = validate_file_path("subdir/data.csv");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/home/user/subdir/data.csv")
        );
    }

    #[test]
    fn test_validate_file_path_deeply_nested() {
        let result = validate_file_path("a/b/c/d/e.txt");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_file_path_traversal_rejected() {
        let result = validate_file_path("../etc/passwd");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("traversal"));
    }

    #[test]
    fn test_validate_file_path_mid_path_traversal() {
        let result = validate_file_path("subdir/../../etc/shadow");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_file_path_absolute_rejected() {
        let result = validate_file_path("/etc/passwd");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Absolute"));
    }

    #[test]
    fn test_validate_file_path_empty_returns_sandbox_root() {
        let result = validate_file_path("");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), std::path::PathBuf::from("/home/user"));
    }

    #[test]
    fn test_validate_file_path_null_byte_rejected() {
        let result = validate_file_path("file\x00.txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("null byte"));
    }

    #[test]
    fn test_validate_file_path_control_chars_rejected() {
        let result = validate_file_path("file\x01name");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("control character"));
    }

    #[test]
    fn test_validate_file_path_too_long() {
        let long_path = "a".repeat(256);
        let result = validate_file_path(&long_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too long"));
    }

    #[test]
    fn test_validate_file_path_exactly_max_length() {
        let path = "a".repeat(255);
        let result = validate_file_path(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_file_path_dot_dot_normalization() {
        // subdir/../file.txt should resolve to /home/user/file.txt (stays under root)
        let result = validate_file_path("subdir/../file.txt");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/home/user/file.txt")
        );
    }

    #[test]
    fn test_validate_file_path_double_slash() {
        let result = validate_file_path("dir//file.txt");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_file_path_trailing_slash() {
        let result = validate_file_path("subdir/");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_file_path_hidden_file() {
        let result = validate_file_path(".hidden");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/home/user/.hidden")
        );
    }

    #[test]
    fn test_validate_file_path_unicode() {
        let result = validate_file_path("日本語.txt");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_file_path_dot_only() {
        // "." should resolve to sandbox root
        let result = validate_file_path(".");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), std::path::PathBuf::from("/home/user"));
    }

    // -------------------------------------------------------------------------
    // GuestCommand deserialization tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_deserialize_write_file() {
        let json =
            r#"{"action":"write_file","op_id":"abc123","path":"test.txt","make_executable":false}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::WriteFile {
                op_id,
                path,
                make_executable,
            } => {
                assert_eq!(op_id, "abc123");
                assert_eq!(path, "test.txt");
                assert!(!make_executable);
            }
            _ => panic!("Expected WriteFile"),
        }
    }

    #[test]
    fn test_deserialize_read_file() {
        let json = r#"{"action":"read_file","op_id":"def456","path":"output.csv"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::ReadFile { op_id, path } => {
                assert_eq!(op_id, "def456");
                assert_eq!(path, "output.csv");
            }
            _ => panic!("Expected ReadFile"),
        }
    }

    #[test]
    fn test_deserialize_list_files() {
        let json = r#"{"action":"list_files","path":""}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::ListFiles { path } => {
                assert_eq!(path, "");
            }
            _ => panic!("Expected ListFiles"),
        }
    }

    #[test]
    fn test_deserialize_list_files_default_path() {
        let json = r#"{"action":"list_files"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::ListFiles { path } => {
                assert_eq!(path, "");
            }
            _ => panic!("Expected ListFiles"),
        }
    }

    #[test]
    fn test_deserialize_write_file_executable() {
        let json =
            r#"{"action":"write_file","op_id":"xyz","path":"run.sh","make_executable":true}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::WriteFile {
                make_executable, ..
            } => {
                assert!(make_executable);
            }
            _ => panic!("Expected WriteFile"),
        }
    }

    #[test]
    fn test_deserialize_file_chunk() {
        let json = r#"{"action":"file_chunk","op_id":"abc123","data":"SGVsbG8="}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::FileChunk { op_id, data } => {
                assert_eq!(op_id, "abc123");
                assert_eq!(data, "SGVsbG8=");
            }
            _ => panic!("Expected FileChunk"),
        }
    }

    #[test]
    fn test_deserialize_file_end() {
        let json = r#"{"action":"file_end","op_id":"abc123"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::FileEnd { op_id } => {
                assert_eq!(op_id, "abc123");
            }
            _ => panic!("Expected FileEnd"),
        }
    }

    // -------------------------------------------------------------------------
    // WarmRepl deserialization tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_deserialize_warm_repl() {
        let json = r#"{"action":"warm_repl","language":"python"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::WarmRepl { language } => {
                assert_eq!(language, "python");
            }
            _ => panic!("Expected WarmRepl"),
        }
    }

    #[test]
    fn test_deserialize_warm_repl_javascript() {
        let json = r#"{"action":"warm_repl","language":"javascript"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::WarmRepl { language } => {
                assert_eq!(language, "javascript");
            }
            _ => panic!("Expected WarmRepl"),
        }
    }

    #[test]
    fn test_deserialize_warm_repl_unknown_language() {
        // Guest accepts any string — validation is at host side
        let json = r#"{"action":"warm_repl","language":"cobol"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::WarmRepl { language } => {
                assert_eq!(language, "cobol");
            }
            _ => panic!("Expected WarmRepl"),
        }
    }

    #[test]
    fn test_deserialize_warm_repl_empty_language() {
        let json = r#"{"action":"warm_repl","language":""}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::WarmRepl { language } => {
                assert_eq!(language, "");
            }
            _ => panic!("Expected WarmRepl"),
        }
    }

    #[test]
    fn test_deserialize_warm_repl_missing_language() {
        // Missing required field should fail
        let json = r#"{"action":"warm_repl"}"#;
        let result = serde_json::from_str::<GuestCommand>(json);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Response serialization tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_serialize_file_write_ack() {
        let ack = GuestResponse::FileWriteAck {
            op_id: "test_op".to_string(),
            path: "test.txt".to_string(),
            bytes_written: 42,
        };
        let json = serde_json::to_string(&ack).unwrap();
        assert!(json.contains("\"type\":\"file_write_ack\""));
        assert!(json.contains("\"op_id\":\"test_op\""));
        assert!(json.contains("\"bytes_written\":42"));
    }

    #[test]
    fn test_serialize_file_chunk_response() {
        let chunk = GuestResponse::FileChunk {
            op_id: "abc123".to_string(),
            data: "SGVsbG8=".to_string(),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"type\":\"file_chunk\""));
        assert!(json.contains("\"op_id\":\"abc123\""));
        assert!(json.contains("\"data\":\"SGVsbG8=\""));
    }

    #[test]
    fn test_serialize_file_read_complete() {
        let msg = GuestResponse::FileReadComplete {
            op_id: "def456".to_string(),
            path: "data.csv".to_string(),
            size: 1024,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"file_read_complete\""));
        assert!(json.contains("\"op_id\":\"def456\""));
        assert!(json.contains("\"size\":1024"));
    }

    #[test]
    fn test_zstd_roundtrip() {
        // Compress
        let data = vec![42u8; FILE_TRANSFER_CHUNK_SIZE];
        let compressed = zstd::stream::encode_all(&data[..], FILE_TRANSFER_ZSTD_LEVEL).unwrap();
        // Decompress
        let decompressed = zstd::stream::decode_all(&compressed[..]).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_serialize_file_list() {
        let list = GuestResponse::FileList {
            path: "".to_string(),
            entries: vec![
                FileEntry {
                    name: "file.txt".to_string(),
                    is_dir: false,
                    size: 100,
                },
                FileEntry {
                    name: "subdir".to_string(),
                    is_dir: true,
                    size: 0,
                },
            ],
        };
        let json = serde_json::to_string(&list).unwrap();
        assert!(json.contains("\"type\":\"file_list\""));
        assert!(json.contains("\"file.txt\""));
        assert!(json.contains("\"subdir\""));
    }

    // -------------------------------------------------------------------------
    // prepend_env_vars shebang tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prepend_shebang_wraps_tempfile() {
        let code = "#!/usr/bin/awk -f\nBEGIN{print}";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        assert!(result.contains("mktemp"), "should use mktemp");
        assert!(result.contains("cat >"), "should write via cat heredoc");
        assert!(result.contains("chmod +x"), "should make executable");
        assert!(result.contains("#!/usr/bin/awk -f\nBEGIN{print}"));
    }

    #[test]
    fn test_prepend_shebang_with_env_vars() {
        let code = "#!/bin/sh\necho $FOO";
        let env = HashMap::from([("FOO".to_string(), "bar".to_string())]);
        let result = prepend_env_vars(Language::Raw, code, &env);
        // Env export comes before the tempfile block
        let export_pos = result.find("export FOO='bar'").unwrap();
        let mktemp_pos = result.find("mktemp").unwrap();
        assert!(
            export_pos < mktemp_pos,
            "export should precede mktemp block"
        );
    }

    #[test]
    fn test_prepend_shebang_empty_env() {
        let code = "#!/bin/sh\necho hi";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        // Even with empty env, shebang should still be wrapped
        assert!(result.contains("mktemp"));
        assert!(result.contains("chmod +x"));
    }

    #[test]
    fn test_prepend_no_shebang_passthrough() {
        let code = "echo hello";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        assert_eq!(result, "echo hello");
    }

    #[test]
    fn test_prepend_no_shebang_with_env() {
        let code = "echo $X";
        let env = HashMap::from([("X".to_string(), "1".to_string())]);
        let result = prepend_env_vars(Language::Raw, code, &env);
        assert!(result.contains("export X='1'"));
        assert!(result.contains("echo $X"));
        assert!(!result.contains("mktemp"), "no shebang = no wrapping");
    }

    #[test]
    fn test_prepend_shebang_sentinel_collision() {
        let code = "#!/bin/sh\necho _EXEC_SANDBOX_EOF_";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        // The sentinel must not be the base string since it appears in code
        assert!(result.contains("_EXEC_SANDBOX_EOF_X"));
    }

    #[test]
    fn test_prepend_shebang_sentinel_double_collision() {
        let code = "#!/bin/sh\necho _EXEC_SANDBOX_EOF_ _EXEC_SANDBOX_EOF_X";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        assert!(result.contains("_EXEC_SANDBOX_EOF_XX"));
    }

    #[test]
    fn test_prepend_shebang_non_raw_ignored() {
        let code = "#!/usr/bin/awk -f\nBEGIN{print}";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Python, code, &env);
        assert_eq!(result, code, "non-raw languages should not wrap shebangs");
    }

    #[test]
    fn test_prepend_shebang_quoted_heredoc() {
        let code = "#!/bin/sh\necho $VAR `cmd`";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        // Heredoc sentinel must be single-quoted to prevent shell expansion
        assert!(
            result.contains("<<'_EXEC_SANDBOX_EOF_'"),
            "heredoc delimiter should be single-quoted, got: {}",
            result
        );
    }

    #[test]
    fn test_prepend_shebang_no_leading_whitespace() {
        // The Rust `\<newline>` string continuation strips indentation.
        // Every generated shell line must start at column 0 — especially the
        // heredoc closing sentinel which the shell matches at line start.
        let code = "#!/bin/sh\necho hi";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        for (i, line) in result.lines().enumerate() {
            assert!(
                !line.starts_with(' ') && !line.starts_with('\t'),
                "line {} has leading whitespace: {:?}",
                i,
                line,
            );
        }
    }

    // -------------------------------------------------------------------------
    // validate_execute_params tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_execute_params_null_byte_in_code() {
        let env = HashMap::new();
        let result = validate_execute_params("print('hi')\0print('bye')", 30, &env);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("null bytes"));
    }

    #[test]
    fn test_validate_execute_params_only_null_byte() {
        let env = HashMap::new();
        let result = validate_execute_params("\0", 30, &env);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_execute_params_valid_code_no_null() {
        let env = HashMap::new();
        let result = validate_execute_params("print('hello')", 30, &env);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prepend_shebang_mktemp_guard() {
        let code = "#!/bin/sh\necho hi";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        assert!(
            result.contains("|| {") && result.contains("exit 126"),
            "mktemp should have an error guard, got: {}",
            result,
        );
    }

    // -------------------------------------------------------------------------
    // prepend_env_vars BOM stripping tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_prepend_strips_bom_python() {
        let code = "\u{FEFF}print('hello')";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Python, code, &env);
        assert_eq!(result, "print('hello')");
    }

    #[test]
    fn test_prepend_strips_bom_raw() {
        let code = "\u{FEFF}echo hello";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        assert_eq!(result, "echo hello");
    }

    #[test]
    fn test_prepend_strips_bom_shebang() {
        let code = "\u{FEFF}#!/bin/sh\necho hello";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Raw, code, &env);
        // After BOM stripping, shebang is detected → tempfile path
        assert!(
            result.contains("mktemp"),
            "shebang should be detected after BOM stripping, got: {}",
            result,
        );
    }

    #[test]
    fn test_prepend_no_bom_unchanged() {
        let code = "print('hello')";
        let env = HashMap::new();
        let result = prepend_env_vars(Language::Python, code, &env);
        assert_eq!(result, "print('hello')");
    }
}

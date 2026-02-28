// Security: set PR_SET_DUMPABLE=0 to prevent ptrace from other UID 1000 processes.
// Must be done here (after exec) because begin_new_exec() always resets dumpable
// to 1, regardless of credential state. Blocks CVE-2022-30594 style attacks.
//
// Blocking I/O: libc.read() replaces Bun.stdin.stream() for stdin reads.
// Bun's stream API sets O_NONBLOCK and polls the event loop at ~20-25% CPU
// when idle. Blocking libc.read() lets the thread sleep in the kernel (0% CPU).
// See: oven-sh/bun#10080, #21081, #27365
import { dlopen, FFIType } from 'bun:ffi';
import { createContext, runInContext } from 'node:vm';
import { Readable } from 'node:stream';
import { releaseWeakRefs } from 'bun:jsc';
const libc = dlopen('libc.so.6', {
    prctl: { args: [FFIType.i32, FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64], returns: FFIType.i32 },
    read:  { args: [FFIType.i32, FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
    fcntl: { args: [FFIType.i32, FFIType.i32, FFIType.i32], returns: FFIType.i32 },
});
libc.symbols.prctl(4, 0, 0, 0, 0);  // PR_SET_DUMPABLE=0
// Ensure stdin is blocking — Bun's runtime may set O_NONBLOCK on fd 0.
const _flags = libc.symbols.fcntl(0, 3/*F_GETFL*/, 0);
if (_flags >= 0) libc.symbols.fcntl(0, 4/*F_SETFL*/, _flags & ~2048/*O_NONBLOCK*/);

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
    },
    set(target, prop, value) {
        if (prop === 'stdin') return true;
        target[prop] = value;
        return true;
    }
});
// VM context globals: createContext() only provides ECMAScript built-ins
// (Object, Array, Math, Promise, etc.). All Web API and Node/Bun globals
// must be passed explicitly — they are NOT auto-injected.
// See: https://bun.sh/docs/runtime/globals
const ctx = createContext({
    Bun: sandboxBun,
    console, process: sandboxProcess, setTimeout, setInterval, clearTimeout, clearInterval,
    Buffer, URL, URLSearchParams, TextEncoder, TextDecoder, fetch,
    Request, Response, Headers, Blob, FormData,
    ReadableStream, WritableStream, TransformStream,
    AbortController, AbortSignal, Event, EventTarget,
    WebAssembly,
    atob, btoa, structuredClone, queueMicrotask,
    crypto, performance,
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
// Reclaim initialization garbage before entering idle read loop.
// Saves ~3-8MB RSS that would otherwise persist until first user execution.
releaseWeakRefs();
Bun.gc(true);
Bun.shrink();
// --- Blocking stdin reader via libc.read() ---
// Thread sleeps in kernel until data arrives, achieving 0% idle CPU.
// The original Bun.stdin.stream().getReader() polls at ~20-25% due to
// O_NONBLOCK + event loop scheduling overhead.
let buf = new Uint8Array(0);
const dec = new TextDecoder();
function cat(a, b) {
    const r = new Uint8Array(a.length + b.length);
    r.set(a); r.set(b, a.length);
    return r;
}
const _rdBuf = new Uint8Array(65536);
function _rd() {
    const n = libc.symbols.read(0, _rdBuf, 65536);
    if (n <= 0) return false;  // EOF or error
    buf = cat(buf, _rdBuf.subarray(0, n));
    return true;
}
function readLine() {
    while (true) {
        const idx = buf.indexOf(10);
        if (idx !== -1) {
            const line = dec.decode(buf.slice(0, idx));
            buf = buf.slice(idx + 1);
            return line;
        }
        if (!_rd()) return null;
    }
}
function readN(n) {
    while (buf.length < n) {
        if (!_rd()) return null;
    }
    const data = buf.slice(0, n);
    buf = buf.slice(n);
    return dec.decode(data);
}
while (true) {
    const header = readLine();
    if (header === null) break;
    const sp = header.indexOf(' ');
    const sentinelId = header.substring(0, sp);
    const codeLen = parseInt(header.substring(sp + 1), 10);
    const code = readN(codeLen);
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
    // Flush stdout before sentinel — process.stdout.write() is async for pipes,
    // so large outputs may still be in Bun's internal buffer. Without this,
    // the sentinel can arrive before stdout fully drains (~100KB+ truncation).
    // Python's sys.stdout.flush() is synchronous so doesn't need this.
    // Uses polling instead of the 'drain' event to avoid a race where drain
    // fires between the writableLength check and the listener registration.
    while (process.stdout.writableLength > 0) {
        await new Promise(resolve => setTimeout(resolve, 1));
    }
    process.stderr.write(`__SENTINEL_${sentinelId}_${exitCode}__\n`);
}

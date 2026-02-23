// Security: set PR_SET_DUMPABLE=0 to prevent ptrace from other UID 1000 processes.
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

// Security: set PR_SET_DUMPABLE=0 to prevent ptrace from other UID 1000 processes.
// Must be done here (after exec) because begin_new_exec() always resets dumpable
// to 1, regardless of credential state. Blocks CVE-2022-30594 style attacks.
//
// Blocking I/O: libc.read() replaces Bun.stdin.stream() for stdin reads.
// Bun's stream API sets O_NONBLOCK and polls the event loop at ~20-25% CPU
// when idle. Blocking libc.read() lets the thread sleep in the kernel (0% CPU).
// See: oven-sh/bun#10080, #21081, #27365
import { dlopen, FFIType } from 'bun:ffi';
import { releaseWeakRefs } from 'bun:jsc';
import { Console } from 'node:console';
import { writeSync } from 'node:fs';
import { constants as vmConstants, createContext, runInContext } from 'node:vm';
import { Readable } from 'node:stream';
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
// import() in user code: runInContext uses the main context's module loader,
// with a script name in /home/user so a relative specifier ("./mod.mjs")
// resolves there. The code stays as written: no text rewrite, which would also
// change string literals.
const EXEC_SCRIPT = { filename: '/home/user/<exec>', importModuleDynamically: vmConstants.USE_MAIN_CONTEXT_DEFAULT_LOADER };
// Create a null stdin (immediate EOF) so user code that reads process.stdin
// or Bun.stdin gets EOF instead of blocking on the protocol pipe.
const nullStdin = new Readable({ read() { this.push(null); } });
nullStdin.fd = -1;
nullStdin.isTTY = false;
// The end marker and the sentinel of each exec go through the original streams
// and their original write methods: user code may replace process.stdout,
// process.stderr or their write.
const _stdout = process.stdout;
const _stderr = process.stderr;
const _origStdoutWrite = process.stdout.write;
const _origStderrWrite = process.stderr.write;
// Bun.write(Bun.stdout, data) writes natively, outside the process.stdout
// stream: its pending writes are awaited before the end marker. Only pending
// ones are kept, so a long exec that streams through Bun.write keeps no list
// of every write it made. The wrapper replaces Bun.write itself, so
// require('bun').write goes through it too. The caller gets a promise
// that settles as Bun's: a rejection it leaves unhandled stays unhandled.
const _stdoutBunWrites = new Set();
const _origBunWrite = Bun.write.bind(Bun);
Bun.write = function write(dest, ...args) {
    const result = _origBunWrite(dest, ...args);
    if (dest !== Bun.stdout) return result;
    _stdoutBunWrites.add(result);
    return result.finally(() => { _stdoutBunWrites.delete(result); });
};
// Route console through process.stdout, so that the output of ALL console
// methods (log, info, dir, table, etc.) goes out before the end marker. Bun's
// native console writes directly to fd 1 via syscall, outside the Node.js
// stream layer, with no order against the marker.
const sandboxConsole = new Console({ stdout: process.stdout, stderr: process.stderr });
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
    console: sandboxConsole, process: sandboxProcess, setTimeout, setInterval, clearTimeout, clearInterval,
    Buffer, URL, URLSearchParams, TextEncoder, TextDecoder, fetch,
    Request, Response, Headers, Blob, FormData,
    ReadableStream, WritableStream, TransformStream,
    AbortController, AbortSignal, Event, EventTarget,
    WebAssembly,
    atob, btoa, structuredClone, queueMicrotask,
    crypto, performance,
    require,
    module: { exports: {} },
    exports: {},
    __filename: '<exec>', __dirname: '/tmp',
});
// Math.random: JSC keeps its state in the process, so an L1 template (or a saved
// instance loaded twice) would replay one stream in every VM. Feed it from
// crypto.getRandomValues instead (kernel CRNG, reseeded before every command),
// in both realms: user code reaches the outer one through require'd modules.
// The pool is dropped before each exec, so no buffered value survives a save.
const randomPool = new Uint32Array(1024);
let randomNext = randomPool.length;
const secureRandom = function random() {
    if (randomNext === randomPool.length) {
        crypto.getRandomValues(randomPool);
        randomNext = 0;
    }
    const hi = randomPool[randomNext++] >>> 5;
    const lo = randomPool[randomNext++] >>> 6;
    return (hi * 67108864 + lo) / 9007199254740992; // 53-bit double in [0, 1)
};
Math.random = secureRandom;
runInContext('Math', ctx).random = secureRandom;
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
// A saved stream that user code ended takes no more writes; its queued writes go
// out first (writableFinished, or closed after an error). Polled, not listened:
// user code may remove listeners, and more listeners warn. No time
// limit, as for a write: a slow reader must get all of them before the
// marker; the exec timeout bounds the wait.
async function drained(stream) {
    while (!stream.writableFinished && !stream.closed) {
        await Bun.sleep(1);
    }
}
// Write `text` to a saved stream with its original `write`; resolve when the write
// calls back. No time limit: the agent reads both pipes until the sentinel (past
// the output limit too), so the write ends; with a limit, the REPL would block in
// the read of the next command while Bun still held the write, and the marker or
// the sentinel would come late or never. A stream that user code ended: the text
// goes to its fd once the queued writes are out.
async function protocolWrite(stream, write, fd, text) {
    if (!stream.writableEnded && !stream.destroyed) {
        return new Promise(resolve => write.call(stream, text, () => resolve()));
    }
    await drained(stream);
    let data = Buffer.from(text);
    while (data.length > 0) {
        try {
            data = data.subarray(writeSync(fd, data));
        } catch (e) {
            if (e.code !== 'EAGAIN') return;
            await Bun.sleep(1);
        }
    }
}
// User errors go to the saved stderr: user code may replace process.stderr or
// its write. Their text cannot fail: user code can throw anything, such
// as an object with no prototype, or getters that throw.
function describe(e) {
    try {
        return e && e.stack ? String(e.stack) : String(e);
    } catch {}
    try {
        return Bun.inspect(e);
    } catch {}
    try {
        return Object.prototype.toString.call(e);
    } catch {}
    return 'Error: the thrown value cannot be shown';
}
function writeError(e) {
    return protocolWrite(_stderr, _origStderrWrite, 2, describe(e) + '\n');
}
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
        if (transformed.length > 0) {
            randomNext = randomPool.length;
            let val = runInContext(transformed, ctx, EXEC_SCRIPT);
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
        await writeError(e);
        exitCode = 1;
    }
    // Bun.write(Bun.stdout) writes of this exec end before its marker. One whose
    // source never ends keeps the exec running to its timeout, as an endless loop
    // does: its output would go into the next exec otherwise.
    if (_stdoutBunWrites.size > 0) {
        await Promise.allSettled([..._stdoutBunWrites]);
    }
    // One turn of the event loop (about 15 us), so a rejection this exec left
    // unhandled is reported in this exec, not in the next one.
    await new Promise(resolve => setImmediate(resolve));
    // Check for unhandled rejections from non-last-expression promises
    if (unhandledRejection !== null) {
        await writeError(unhandledRejection);
        exitCode = 1;
    }
    // process.stdout and process.stderr writes are async for pipes: a large
    // output can still be in Bun's buffer here. The read of the next command
    // blocks the event loop, so each write below is awaited until written
    // (with a fixed wait, the sentinel can stay behind a large stderr
    // output). Writes of one stream go out in order: when the
    // end marker is written, all stdout before it is in the pipe.
    // End of this exec's stdout: the agent stops reading stdout here, without
    // waiting for the pipe to go idle.
    await protocolWrite(_stdout, _origStdoutWrite, 1, `__STDOUT_END_${sentinelId}__\n`);
    await protocolWrite(_stderr, _origStderrWrite, 2, `__SENTINEL_${sentinelId}_${exitCode}__\n`);
}

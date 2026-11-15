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
import { Readable } from 'node:stream';
// node:vm through require: an ESM namespace of it made before the patch of its functions
// (Math.random, below) keeps the originals
const nodeVm = require('node:vm');
const { constants: vmConstants, runInContext } = nodeVm;
const libc = dlopen('libc.so.6', {
    prctl: { args: [FFIType.i32, FFIType.u64, FFIType.u64, FFIType.u64, FFIType.u64], returns: FFIType.i32 },
    read:  { args: [FFIType.i32, FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
    fcntl: { args: [FFIType.i32, FFIType.i32, FFIType.i32], returns: FFIType.i32 },
    getrandom: { args: [FFIType.ptr, FFIType.u64, FFIType.u32], returns: FFIType.i32 },
});
if (libc.symbols.prctl(4, 0, 0, 0, 0) !== 0) {  // PR_SET_DUMPABLE=0: fail closed, as the Python REPL
    throw new Error('prctl(PR_SET_DUMPABLE, 0) failed');
}
// Ensure stdin is blocking — Bun's runtime may set O_NONBLOCK on fd 0.
const _flags = libc.symbols.fcntl(0, 3/*F_GETFL*/, 0);
if (_flags >= 0) libc.symbols.fcntl(0, 4/*F_SETFL*/, _flags & ~2048/*O_NONBLOCK*/);

// --- Random values from the kernel ---
// The random APIs below take their bytes from the kernel (getrandom(2)) at each call,
// not from the state that Bun's entropy cache and BoringSSL's generators keep in the
// memory of this process. Each wrapper first calls Bun's original (randomInt: its sync
// form, with no callback), so argument checks, errors and return types stay Bun's own,
// and a callback runs when Bun's would; then it puts bytes from the kernel in place of
// the random bytes of the result (randomUUID: a new v4 UUID; randomInt: a new value).
// They are installed here, before any user code, through `require` and before any ESM
// import of node:crypto: an ESM namespace made before the patch keeps the originals. Key
// generation, signatures, TLS and fetch inside Bun, and Workers, keep Bun's own
// generators.
function fillFromKernel(bytes) {
    // bytes: a view or an ArrayBuffer (bun:ffi passes its start). getrandom(2) can return fewer
    // bytes than asked when a signal is pending: then a view of the rest
    const length = bytes.byteLength;
    let done = 0;
    while (done < length) {
        const rest = done === 0 ? bytes : new Uint8Array(bytes.buffer ?? bytes, (bytes.byteOffset ?? 0) + done, length - done);
        const n = libc.symbols.getrandom(rest, length - done, 0);
        if (n <= 0) throw new Error('getrandom(2) failed');
        done += n;
    }
    return bytes;
}
// owner[name] becomes makeWrapper(original), with the name, length and property
// attributes of the original; returns the original. Fails closed (the REPL does not
// start) when the original is not an own function or the property keeps its value: a
// Bun release can change them.
function replace(owner, name, makeWrapper) {
    const descriptor = Object.getOwnPropertyDescriptor(owner, name);
    if (typeof descriptor?.value !== 'function') throw new Error(`kernel random: ${name} is not a function`);
    const original = descriptor.value;
    const wrapper = makeWrapper(original);
    Object.defineProperty(wrapper, 'name', { value: original.name });
    Object.defineProperty(wrapper, 'length', { value: original.length });
    Object.defineProperty(owner, name, { value: wrapper }); // the property keeps its other attributes
    if (owner[name] !== wrapper) throw new Error(`kernel random: ${name} cannot be replaced`);
    return original;
}
// Returns Bun's own randomFillSync (for reseedBoringSsl, below).
function installKernelRandom(webCrypto, nodeCrypto, deprecate) {
    const bytesOf = (value) => (ArrayBuffer.isView(value)
        ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength)
        : new Uint8Array(value));
    // The bytes that Bun's randomFill(Sync) fills: offset and size count the elements of
    // a TypedArray, the bytes of anything else
    function filledRange(buf, offset, size) {
        const all = bytesOf(buf);
        const element = buf.BYTES_PER_ELEMENT ?? 1;
        const start = offset === undefined ? 0 : Math.trunc(offset * element);
        const length = size === undefined ? all.length - start : Math.trunc(size * element);
        return all.subarray(start, start + length);
    }
    // A table built by a loop: Array.from with toString(16) and padStart takes 6 ms at
    // its first run in the guest
    const digits = '0123456789abcdef';
    const hex = [];
    for (let i = 0; i < 256; i++) hex.push(digits[i >> 4] + digits[i & 15]);
    const uuidText = (b) => hex[b[0]] + hex[b[1]] + hex[b[2]] + hex[b[3]] + '-' + hex[b[4]] + hex[b[5]]
        + '-' + hex[b[6]] + hex[b[7]] + '-' + hex[b[8]] + hex[b[9]] + '-' + hex[b[10]] + hex[b[11]]
        + hex[b[12]] + hex[b[13]] + hex[b[14]] + hex[b[15]];
    function uuid4() {
        const b = fillFromKernel(new Uint8Array(16));
        b[6] = (b[6] & 0x0f) | 0x40;
        b[8] = (b[8] & 0x3f) | 0x80;
        return uuidText(b);
    }
    // UUIDv7: the timestamp and the 12-bit rand_a counter (Bun's monotonic state, RFC 9562
    // §6.2) stay; rand_b (bytes 8-15 but the 2 variant bits) comes from the kernel
    function newRandB(bytes) {
        const fresh = fillFromKernel(new Uint8Array(8));
        bytes[8] = (bytes[8] & 0xc0) | (fresh[0] & 0x3f);
        bytes.set(fresh.subarray(1), 9);
        return bytes;
    }
    // The same on a v7 text: its first 19 characters stay (no parse of the rest)
    function withRandB(text) {
        const fresh = fillFromKernel(new Uint8Array(8));
        const b8 = (parseInt(text.slice(19, 21), 16) & 0xc0) | (fresh[0] & 0x3f);
        return text.slice(0, 19) + hex[b8] + hex[fresh[1]] + '-' + hex[fresh[2]] + hex[fresh[3]] + hex[fresh[4]]
            + hex[fresh[5]] + hex[fresh[6]] + hex[fresh[7]];
    }
    // globalThis.crypto of this realm and of the exec context (the context gets this
    // object), require('crypto').webcrypto (the same object) and require('crypto')
    // .getRandomValues (a call of this one)
    const proto = Object.getPrototypeOf(webCrypto);
    replace(proto, 'getRandomValues', (original) => function getRandomValues(...args) {
        const result = Reflect.apply(original, this, args);
        fillFromKernel(result);
        return result;
    });
    replace(proto, 'randomUUID', (original) => function randomUUID(...args) {
        Reflect.apply(original, this, args); // Bun's check of `this`
        return uuid4();
    });
    // A callback of Bun's that gets its result once `fill` put the kernel's bytes in it: a
    // kernel failure goes to the callback as its error, as an error of Bun's does
    const refilled = (callback, fill) => (error, result) => {
        if (!error) {
            try {
                fill(result);
            } catch (failure) {
                callback(failure);
                return;
            }
        }
        callback(error, result);
    };
    replace(nodeCrypto, 'randomBytes', (original) => function randomBytes(...args) {
        const [size, callback] = args;
        if (typeof callback !== 'function') {
            const result = Reflect.apply(original, this, args); // the sync form, or Bun throws
            fillFromKernel(result);
            return result;
        }
        return Reflect.apply(original, this, [size, refilled(callback, (result) => fillFromKernel(result))]);
    });
    // Bun's deprecated aliases hold its original randomBytes: each becomes, as in Bun, a
    // deprecate() of the wrapper above
    for (const name of ['pseudoRandomBytes', 'prng', 'rng']) {
        replace(nodeCrypto, name, () => deprecate(nodeCrypto.randomBytes, `crypto.${name} is deprecated.`, 'DEP0115'));
    }
    const bunRandomFillSync = replace(nodeCrypto, 'randomFillSync', (original) => function randomFillSync(...args) {
        const result = Reflect.apply(original, this, args);
        const [buf, offset, size] = args;
        fillFromKernel(filledRange(buf, offset, size));
        return result;
    });
    replace(nodeCrypto, 'randomFill', (original) => function randomFill(...args) {
        const [buf, offset, size] = args;
        const at = typeof offset === 'function' ? 1 : typeof size === 'function' ? 2 : 3; // the callback's index
        const callback = args[at];
        if (typeof callback !== 'function') return Reflect.apply(original, this, args); // Bun throws
        const fill = () => fillFromKernel(filledRange(buf, ...args.slice(1, at))); // offset and size, as given
        return Reflect.apply(original, this, [...args.slice(0, at), refilled(callback, fill)]);
    });
    // Node's randomInt: rejection sampling over 48-bit values (lib/internal/crypto/random.js)
    const RAND_MAX = 0xffff_ffff_ffff;
    const six = new Uint8Array(6); // one buffer for all draws: randomInt never runs twice at once
    replace(nodeCrypto, 'randomInt', (original) => function randomInt(...args) {
        let [min, max, callback] = args;
        if (max === undefined || typeof max === 'function') {
            callback = max;
            max = min;
            min = 0;
        }
        // Bun's checks and errors (without the callback: Bun's sync form schedules nothing; a
        // callback that is not a function: Bun throws)
        Reflect.apply(original, this, typeof callback === 'function' ? [min, max] : args);
        const range = max - min;
        const limit = RAND_MAX - (RAND_MAX % range);
        let x;
        do {
            fillFromKernel(six);
            x = (six[0] * 256 + six[1]) * 2 ** 32 + (((six[2] << 24) | (six[3] << 16) | (six[4] << 8) | six[5]) >>> 0);
        } while (x >= limit);
        const n = min + (x % range);
        if (callback === undefined) return n;
        process.nextTick(callback, undefined, n);
        return undefined;
    });
    replace(nodeCrypto, 'randomUUID', (original) => function randomUUID(...args) {
        Reflect.apply(original, this, args); // Bun's checks of the options
        return uuid4();
    });
    replace(nodeCrypto, 'randomUUIDv7', (original) => function randomUUIDv7(...args) {
        return withRandB(Reflect.apply(original, this, args));
    });
    replace(Bun, 'randomUUIDv7', (original) => function randomUUIDv7(...args) {
        const result = Reflect.apply(original, this, args);
        if (typeof result !== 'string') return newRandB(bytesOf(result)) && result; // 'buffer'
        const encoding = args[0];
        if (encoding === 'base64' || encoding === 'base64url') {
            return Buffer.from(newRandB(new Uint8Array(Buffer.from(result, encoding)))).toString(encoding);
        }
        return withRandB(result);
    });
    return bunRandomFillSync;
}
const bunRandomFillSync = installKernelRandom(crypto, require('node:crypto'), require('node:util').deprecate);
fillFromKernel(new Uint8Array(1)); // fail closed: the REPL does not start without getrandom(2)
// BoringSSL reseeds the generator of a thread from the kernel after 4096 calls
// (kReseedInterval): 4096 calls of Bun's own randomFillSync on one byte (about 1.1 ms)
// make the generator of this thread take a new seed, used by the keys, signatures and
// TLS of this thread. The generators of Workers and of Bun's HTTP client thread (fetch)
// keep their state.
const reseedByte = new Uint8Array(1);
function reseedBoringSsl() {
    for (let i = 0; i < 4096; i++) bunRandomFillSync(reseedByte);
}

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
// One Proxy handler for process and Bun: it intercepts stdin access, forwarding
// everything else; its set trap keeps user code from replacing the real stdin.
// Bun needs it too: else user code could bypass the process.stdin redirect via
// Bun.stdin.stream().
const noStdin = {
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
};
const sandboxProcess = new Proxy(process, noStdin);
const sandboxBun = new Proxy(Bun, noStdin);
// Math.random: from the kernel (getrandom(2)), not from the generator state that JSC
// keeps in the memory of this process: in this realm (user code reaches it through
// require'd modules), the exec context and each realm that user code makes (below). The
// pool is dropped before each exec, so no value of an earlier exec is used.
const randomPool = new Uint32Array(1024);
const randomPoolBytes = new Uint8Array(randomPool.buffer);
let randomNext = randomPool.length;
function secureRandom() {
    if (randomNext === randomPool.length) {
        fillFromKernel(randomPoolBytes);
        randomNext = 0;
    }
    const hi = randomPool[randomNext++] >>> 5;
    const lo = randomPool[randomNext++] >>> 6;
    return (hi * 67108864 + lo) / 9007199254740992; // 53-bit double in [0, 1)
}
// JSC seeds the generator of each new realm from a state in the memory of this process.
// `install`, run in a realm, gives its Math a random of that realm that calls `source`
// (secureRandom): a function of this realm would give the code there this realm's
// Function, and so Bun and process. It also makes each ShadowRealm made there run
// `install` before its first evaluate or importValue. It runs in this realm and, once, in
// each node:vm context (the exec context too): createContext and Script#runInNewContext
// give it to their context (Bun's runInNewContext calls Script#runInNewContext with the
// context it made). A caller's context object that has a Math or ShadowRealm (own or
// inherited) keeps it. Script#runInNewContext runs as Node defines it (lib/vm.js): the
// caller's object becomes the context. Only the originals taken here run: user code may
// replace the node:vm functions. Workers keep JSC's generator.
const kernelRandom = new nodeVm.Script(`(function install(source, context) {
    'use strict';
    const apply = Reflect.apply;
    // A Math or ShadowRealm of the caller's context object stays (a DONT_CONTEXTIFY context is
    // its own global; no context: this realm, or a ShadowRealm)
    const realms = (name) => context === undefined || context === globalThis || !(name in context);
    if (realms('Math')) {
        Math.random = { random() {
            try {
                return source();
            } catch (error) {
                throw new Error(String(error?.message ?? error)); // an error of this realm
            }
        } }.random;
    }
    if (!realms('ShadowRealm') || typeof ShadowRealm !== 'function') return;
    const text = '(' + install + ')';
    const proto = ShadowRealm.prototype;
    const { evaluate, importValue } = proto;
    const ready = new WeakSet();
    function first(realm, original, args) {
        if (!ready.has(realm)) {
            const installThere = apply(evaluate, realm, [text]); // the check of this: evaluate's own
            ready.add(realm);
            installThere(source);
        }
        return apply(original, realm, args);
    }
    const wrappers = {
        evaluate(sourceText) { return first(this, evaluate, [sourceText]); },
        importValue(specifier, bindingName) { return first(this, importValue, [specifier, bindingName]); },
    };
    for (const name of ['evaluate', 'importValue']) {
        Object.defineProperty(proto, name, { value: wrappers[name] }); // the property keeps its other attributes
    }
})`);
kernelRandom.runInThisContext()(secureRandom, undefined);
const scriptRunInContext = nodeVm.Script.prototype.runInContext;
// Once per context: code there may have set its own Math.random since, and a second install
// would run on intrinsics that this code may have replaced (Reflect.apply): they would get
// secureRandom, a function of this realm.
const kernelRandomContexts = new WeakSet();
function withKernelRandom(context) {
    if (kernelRandomContexts.has(context)) return context;
    Reflect.apply(scriptRunInContext, kernelRandom, [context])(secureRandom, context);
    kernelRandomContexts.add(context);
    return context;
}
const bunCreateContext = replace(nodeVm, 'createContext', (original) => function createContext(...args) {
    return withKernelRandom(Reflect.apply(original, this, args));
});
replace(nodeVm.Script.prototype, 'runInNewContext', () => function runInNewContext(contextObject, options) {
    // The context options of a run, as Node's getContextOptions maps them
    const contextOptions = options !== null && typeof options === 'object' ? {
        name: options.contextName,
        origin: options.contextOrigin,
        codeGeneration: options.contextCodeGeneration,
        microtaskMode: options.microtaskMode,
    } : undefined;
    const context = withKernelRandom(bunCreateContext(contextObject, contextOptions));
    return Reflect.apply(scriptRunInContext, this, [context, options]);
});
// VM context globals: createContext() only provides ECMAScript built-ins
// (Object, Array, Math, Promise, etc.). All Web API and Node/Bun globals
// must be passed explicitly — they are NOT auto-injected.
// See: https://bun.sh/docs/runtime/globals
const ctx = nodeVm.createContext({
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
// Bun.stdin.stream().getReader() polls at ~20-25% due to O_NONBLOCK +
// event loop scheduling overhead.
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
// Ready: the agent writes the first exec only after it reads this line (a REPL that
// fails before it did not start, and no user code ran)
await protocolWrite(_stderr, _origStderrWrite, 2, '__REPL_READY__\n');
while (true) {
    const header = readLine();
    if (header === null) break;
    // reseed: '1' to reseed BoringSSL's generator of this thread before the code runs
    const [sentinelId, codeLenText, reseed] = header.split(' ');
    const code = readN(parseInt(codeLenText, 10));
    if (code === null) break;
    let exitCode = 0;
    unhandledRejection = null;
    try {
        // First: a failure shows as this exec's error, and the code does not run
        if (reseed === '1') reseedBoringSsl();
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

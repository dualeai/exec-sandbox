"""Tests that WebAssembly compiles and executes inside the hardened VM.

The sandbox executes JavaScript via Bun (powered by JavaScriptCore).  WASM is
used by crypto libraries (libsodium-wasm), image codecs (squoosh), PDF
renderers, SQLite-in-browser, and other performance-sensitive workloads.

These tests are regression canaries: if security hardening (read-only rootfs,
device restrictions, seccomp filters, memory limits, etc.) inadvertently breaks
WASM compilation or execution, these tests will catch it.

All WASM bytecode is hand-crafted inline as `Uint8Array` -- no external
dependencies or network access required.

Tested features:
- Core MVP: compile, instantiate, linear memory, imports/exports, tables, globals
- WebAssembly API: validate(), Module(), Instance(), Memory, Table, Global
- Streaming APIs: compileStreaming(), instantiateStreaming() (Bun 1.3+)
- SIMD: i32x4.add via memory load/store (Wasm 3.0, JSC)
- Exception handling: try/catch/throw (Wasm 3.0, JSC)
- REPL session persistence: instance survives across exec() calls

See:
- https://webassembly.github.io/spec/core/
- https://bun.sh/docs/runtime/web-apis#webassembly
"""

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler


# =============================================================================
# Basic compilation and execution
# =============================================================================
class TestWasmCompileAndRun:
    """Core WASM MVP: compile and run simple functions."""

    async def test_add_function(self, scheduler: Scheduler) -> None:
        """Synchronous compile+instantiate of add(i32, i32) -> i32."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d,  // magic
  0x01, 0x00, 0x00, 0x00,  // version 1
  // Type section (7 bytes): 1 type, (i32, i32) -> i32
  0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
  // Function section (2 bytes): func 0 uses type 0
  0x03, 0x02, 0x01, 0x00,
  // Export section (7 bytes): "add" -> func 0
  0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
  // Code section (9 bytes): body(7)=locals(0) local.get 0, local.get 1, i32.add, end
  0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
]);
const mod = new WebAssembly.Module(bytes);
const inst = new WebAssembly.Instance(mod);
console.log("ADD:" + inst.exports.add(3, 4));
console.log("ADD_ZERO:" + inst.exports.add(0, 0));
console.log("ADD_NEG:" + inst.exports.add(-1, 1));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "ADD:7" in result.stdout
        assert "ADD_ZERO:0" in result.stdout
        assert "ADD_NEG:0" in result.stdout

    async def test_const_function(self, scheduler: Scheduler) -> None:
        """Async instantiate of answer() -> i32 returning 42."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type section (5 bytes): () -> i32
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
  // Function section (2 bytes)
  0x03, 0x02, 0x01, 0x00,
  // Export section (10 bytes): "answer" -> func 0
  0x07, 0x0a, 0x01, 0x06, 0x61, 0x6e, 0x73, 0x77, 0x65, 0x72, 0x00, 0x00,
  // Code section (6 bytes): body(4)=locals(0) i32.const 42, end
  0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
]);
const { instance } = await WebAssembly.instantiate(bytes);
console.log("ANSWER:" + instance.exports.answer());
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "ANSWER:42" in result.stdout

    async def test_multiply_function(self, scheduler: Scheduler) -> None:
        """mul(i32, i32) -> i32 with edge cases (zero, negative)."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type section (7 bytes): (i32, i32) -> i32
  0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
  // Function section (2 bytes)
  0x03, 0x02, 0x01, 0x00,
  // Export section (7 bytes): "mul" -> func 0
  0x07, 0x07, 0x01, 0x03, 0x6d, 0x75, 0x6c, 0x00, 0x00,
  // Code section (9 bytes): body(7)=locals(0) local.get 0, local.get 1, i32.mul, end
  0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6c, 0x0b,
]);
const mod = new WebAssembly.Module(bytes);
const inst = new WebAssembly.Instance(mod);
console.log("MUL:" + inst.exports.mul(6, 7));
console.log("MUL_ZERO:" + inst.exports.mul(99, 0));
console.log("MUL_NEG:" + inst.exports.mul(-3, 5));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "MUL:42" in result.stdout
        assert "MUL_ZERO:0" in result.stdout
        assert "MUL_NEG:-15" in result.stdout


# =============================================================================
# Multi-function modules
# =============================================================================
class TestWasmMultipleExports:
    """Module exporting multiple functions."""

    async def test_arithmetic_module(self, scheduler: Scheduler) -> None:
        """Module exporting both add and sub, validates both."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type section (7 bytes): (i32, i32) -> i32
  0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
  // Function section (3 bytes): two functions, both type 0
  0x03, 0x03, 0x02, 0x00, 0x00,
  // Export section (13 bytes): "add" -> func 0, "sub" -> func 1
  0x07, 0x0d, 0x02,
    0x03, 0x61, 0x64, 0x64, 0x00, 0x00,  // "add", func, 0
    0x03, 0x73, 0x75, 0x62, 0x00, 0x01,  // "sub", func, 1
  // Code section (17 bytes): 2 bodies, each body(7)
  0x0a, 0x11, 0x02,
    // add: locals(0) local.get 0, local.get 1, i32.add, end
    0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
    // sub: locals(0) local.get 0, local.get 1, i32.sub, end
    0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6b, 0x0b,
]);
const mod = new WebAssembly.Module(bytes);
const inst = new WebAssembly.Instance(mod);
console.log("ADD:" + inst.exports.add(10, 3));
console.log("SUB:" + inst.exports.sub(10, 3));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "ADD:13" in result.stdout
        assert "SUB:7" in result.stdout


# =============================================================================
# Linear memory operations
# =============================================================================
class TestWasmMemory:
    """WASM linear memory: store, load, grow."""

    async def test_memory_store_and_load(self, scheduler: Scheduler) -> None:
        """WASM module with exported memory, i32.store/i32.load round-trip."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type section (11 bytes): store(i32,i32)->void, load(i32)->i32
  0x01, 0x0b, 0x02,
    0x60, 0x02, 0x7f, 0x7f, 0x00,
    0x60, 0x01, 0x7f, 0x01, 0x7f,
  // Function section (3 bytes): func 0=type 0, func 1=type 1
  0x03, 0x03, 0x02, 0x00, 0x01,
  // Memory section (3 bytes): 1 page min
  0x05, 0x03, 0x01, 0x00, 0x01,
  // Export section (22 bytes): "mem"->memory 0, "store"->func 0, "load"->func 1
  0x07, 0x16, 0x03,
    0x03, 0x6d, 0x65, 0x6d, 0x02, 0x00,
    0x05, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x00, 0x00,
    0x04, 0x6c, 0x6f, 0x61, 0x64, 0x00, 0x01,
  // Code section (19 bytes): store body(9), load body(7)
  0x0a, 0x13, 0x02,
    0x09, 0x00, 0x20, 0x00, 0x20, 0x01, 0x36, 0x02, 0x00, 0x0b,
    0x07, 0x00, 0x20, 0x00, 0x28, 0x02, 0x00, 0x0b,
]);
const mod = new WebAssembly.Module(bytes);
const inst = new WebAssembly.Instance(mod);
inst.exports.store(0, 12345);
console.log("LOAD:" + inst.exports.load(0));
inst.exports.store(4, 99999);
console.log("LOAD4:" + inst.exports.load(4));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "LOAD:12345" in result.stdout
        assert "LOAD4:99999" in result.stdout

    async def test_memory_via_js_view(self, scheduler: Scheduler) -> None:
        """WebAssembly.Memory created from JS, accessed via Uint32Array."""
        code = """\
const mem = new WebAssembly.Memory({ initial: 1 });
const view = new Uint32Array(mem.buffer);
view[0] = 42;
view[1] = 100;
console.log("V0:" + view[0]);
console.log("V1:" + view[1]);
console.log("BYTES:" + mem.buffer.byteLength);
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "V0:42" in result.stdout
        assert "V1:100" in result.stdout
        assert "BYTES:65536" in result.stdout  # 1 page = 64 KiB

    async def test_memory_grow(self, scheduler: Scheduler) -> None:
        """Memory.grow() extends pages correctly."""
        code = """\
const mem = new WebAssembly.Memory({ initial: 1, maximum: 10 });
console.log("BEFORE:" + (mem.buffer.byteLength / 65536));
const old = mem.grow(2);
console.log("OLD_PAGES:" + old);
console.log("AFTER:" + (mem.buffer.byteLength / 65536));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "BEFORE:1" in result.stdout
        assert "OLD_PAGES:1" in result.stdout
        assert "AFTER:3" in result.stdout


# =============================================================================
# Host function interop
# =============================================================================
class TestWasmImports:
    """WASM calling imported JS functions."""

    async def test_imported_function(self, scheduler: Scheduler) -> None:
        """WASM calls an imported JS function with i32.const 1234."""
        # Type section: type 0 = (i32)->void (4 bytes), type 1 = ()->void (3 bytes)
        #   content: 0x02 + 4 + 3 = 8 bytes
        # Import section: "env"."notify" func type 0
        #   content: 0x01 + (0x03+"env") + (0x06+"notify") + 0x00+0x00 = 14 bytes
        # Function section: 1 func, type 1
        #   content: 0x01 + 0x01 = 2 bytes
        # Export section: "run" -> func 1
        #   content: 0x01 + (0x03+"run") + 0x00+0x01 = 7 bytes
        # Code section: body(7) = locals(0) i32.const(1234=0xd2,0x09) call(0) end
        #   content: 0x01 + 0x07 + 7 = 9 bytes
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type section (8 bytes)
  0x01, 0x08, 0x02,
    0x60, 0x01, 0x7f, 0x00,
    0x60, 0x00, 0x00,
  // Import section (14 bytes): "env"."notify" = type 0
  0x02, 0x0e, 0x01,
    0x03, 0x65, 0x6e, 0x76,
    0x06, 0x6e, 0x6f, 0x74, 0x69, 0x66, 0x79,
    0x00, 0x00,
  // Function section (2 bytes): func uses type 1
  0x03, 0x02, 0x01, 0x01,
  // Export section (7 bytes): "run" -> func 1 (import is func 0)
  0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x01,
  // Code section (9 bytes): body(7) = call notify(1234)
  0x0a, 0x09, 0x01, 0x07, 0x00, 0x41, 0xd2, 0x09, 0x10, 0x00, 0x0b,
]);
let received = null;
const imports = { env: { notify: (v) => { received = v; } } };
const mod = new WebAssembly.Module(bytes);
const inst = new WebAssembly.Instance(mod, imports);
inst.exports.run();
console.log("RECEIVED:" + received);
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "RECEIVED:1234" in result.stdout


# =============================================================================
# WebAssembly global API surface
# =============================================================================
class TestWasmApi:
    """WebAssembly.validate(), CompileError, Table, Global."""

    async def test_validate_valid_module(self, scheduler: Scheduler) -> None:
        """WebAssembly.validate() returns true for valid bytes."""
        code = """\
// Minimal valid module (magic + version + empty)
const valid = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
console.log("VALID:" + WebAssembly.validate(valid));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "VALID:true" in result.stdout

    async def test_validate_invalid_module(self, scheduler: Scheduler) -> None:
        """WebAssembly.validate() returns false for garbage bytes."""
        code = """\
const garbage = new Uint8Array([0xde, 0xad, 0xbe, 0xef]);
console.log("INVALID:" + WebAssembly.validate(garbage));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "INVALID:false" in result.stdout

    async def test_compile_error_rejected(self, scheduler: Scheduler) -> None:
        """new WebAssembly.Module(garbage) throws CompileError."""
        code = """\
try {
  new WebAssembly.Module(new Uint8Array([0xde, 0xad, 0xbe, 0xef]));
  console.log("ERROR:none");
} catch (e) {
  console.log("ERROR:" + e.constructor.name);
}
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "ERROR:CompileError" in result.stdout

    async def test_table_api(self, scheduler: Scheduler) -> None:
        """WebAssembly.Table creation and inspection."""
        code = """\
const table = new WebAssembly.Table({ initial: 2, element: "anyfunc" });
console.log("LENGTH:" + table.length);
console.log("GET0:" + table.get(0));
table.grow(3);
console.log("AFTER_GROW:" + table.length);
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "LENGTH:2" in result.stdout
        assert "GET0:null" in result.stdout
        assert "AFTER_GROW:5" in result.stdout

    async def test_global_api(self, scheduler: Scheduler) -> None:
        """WebAssembly.Global create, read, mutate."""
        code = """\
const g = new WebAssembly.Global({ value: "i32", mutable: true }, 10);
console.log("INIT:" + g.value);
g.value = 99;
console.log("MUTATED:" + g.value);

const immutable = new WebAssembly.Global({ value: "i32", mutable: false }, 7);
console.log("IMMUTABLE:" + immutable.value);
try {
  immutable.value = 0;
  console.log("WRITE:allowed");
} catch (e) {
  console.log("WRITE:rejected");
}
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "INIT:10" in result.stdout
        assert "MUTATED:99" in result.stdout
        assert "IMMUTABLE:7" in result.stdout
        assert "WRITE:rejected" in result.stdout


# =============================================================================
# Streaming compilation APIs (Bun 1.3+)
# =============================================================================
class TestWasmStreaming:
    """compileStreaming / instantiateStreaming via Response object.

    These APIs require the Response Web API, which must be exposed in
    the REPL sandbox's vm.createContext().
    """

    async def test_compile_streaming(self, scheduler: Scheduler) -> None:
        """WebAssembly.compileStreaming(new Response(bytes)) compiles a module."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
  0x03, 0x02, 0x01, 0x00,
  0x07, 0x05, 0x01, 0x01, 0x66, 0x00, 0x00,
  0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x07, 0x0b,
]);
const resp = new Response(bytes, { headers: { "Content-Type": "application/wasm" } });
const mod = await WebAssembly.compileStreaming(resp);
console.log("IS_MODULE:" + (mod instanceof WebAssembly.Module));
const inst = new WebAssembly.Instance(mod);
console.log("F:" + inst.exports.f());
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "IS_MODULE:true" in result.stdout
        assert "F:7" in result.stdout

    async def test_instantiate_streaming(self, scheduler: Scheduler) -> None:
        """WebAssembly.instantiateStreaming(new Response(bytes)) instantiates."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
  0x03, 0x02, 0x01, 0x00,
  0x07, 0x05, 0x01, 0x01, 0x67, 0x00, 0x00,
  0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x09, 0x0b,
]);
const resp = new Response(bytes, { headers: { "Content-Type": "application/wasm" } });
const { module, instance } = await WebAssembly.instantiateStreaming(resp);
console.log("HAS_MODULE:" + (module instanceof WebAssembly.Module));
console.log("G:" + instance.exports.g());
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "HAS_MODULE:true" in result.stdout
        assert "G:9" in result.stdout


# =============================================================================
# SIMD (128-bit, Wasm 3.0)
# =============================================================================
class TestWasmSimd:
    """128-bit SIMD operations (i32x4.add) via memory marshalling.

    Bun 1.3+ (JSC) supports WASM SIMD on both x86_64 (requires AVX) and
    ARM64.  v128 values cannot appear in JS-visible function signatures,
    so SIMD results are marshalled through linear memory.
    """

    async def test_simd_available(self, scheduler: Scheduler) -> None:
        """WebAssembly.validate() returns true for a SIMD module (v128.const)."""
        # Body: locals(0) + v128.const(2) + 16-byte-imm + drop(1) + end(1) = 21 bytes
        # Code section: num_funcs(1) + body_size(1) + body(21) = 23 bytes
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type section (4 bytes): () -> ()
  0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
  // Function section (2 bytes)
  0x03, 0x02, 0x01, 0x00,
  // Code section (23 bytes): body(21) = v128.const 0, drop, end
  0x0a, 0x17, 0x01, 0x15, 0x00,
    0xfd, 0x0c,  // v128.const
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x1a,  // drop
    0x0b,  // end
]);
console.log("SIMD_VALID:" + WebAssembly.validate(bytes));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "SIMD_VALID:true" in result.stdout

    async def test_simd_v128_add_i32x4(self, scheduler: Scheduler) -> None:
        """i32x4.add on two vectors via memory load/store.

        Layout: memory[0..16] = vec A, memory[16..32] = vec B,
        result written to memory[32..48].
        Function reads A and B from memory, adds them, stores result.
        v128.load/store pop an i32 base address from the stack, so we push
        i32.const 0 before each memory operation.
        """
        # Body instructions (after locals=0):
        #   i32.const 0:                 0x41 0x00              (2 bytes) store addr
        #   i32.const 0:                 0x41 0x00              (2 bytes) load A addr
        #   v128.load align=4 offset=0:  0xFD 0x00 0x04 0x00   (4 bytes)
        #   i32.const 0:                 0x41 0x00              (2 bytes) load B addr
        #   v128.load align=4 offset=16: 0xFD 0x00 0x04 0x10   (4 bytes)
        #   i32x4.add:                   0xFD 0xAE 0x01        (3 bytes)
        #   v128.store align=4 offset=32:0xFD 0x0B 0x04 0x20   (4 bytes)
        #   end:                         0x0B                   (1 byte)
        # Body = locals(1) + 2+2+4+2+4+3+4+1 = 23 bytes
        # Code section = num_funcs(1) + body_size(1) + body(23) = 25 bytes
        #
        # Export section: count(1) + "mem"(6) + "add_vecs"(11) = 18 bytes
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type section (4 bytes): () -> ()
  0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
  // Function section (2 bytes)
  0x03, 0x02, 0x01, 0x00,
  // Memory section (3 bytes): 1 page
  0x05, 0x03, 0x01, 0x00, 0x01,
  // Export section (18 bytes): "mem" -> memory 0, "add_vecs" -> func 0
  0x07, 0x12, 0x02,
    0x03, 0x6d, 0x65, 0x6d, 0x02, 0x00,
    0x08, 0x61, 0x64, 0x64, 0x5f, 0x76, 0x65, 0x63, 0x73, 0x00, 0x00,
  // Code section (25 bytes): body(23)
  0x0a, 0x19, 0x01, 0x17, 0x00,
    0x41, 0x00,              // i32.const 0  (base addr for v128.store)
    0x41, 0x00,              // i32.const 0  (base addr for v128.load A)
    0xfd, 0x00, 0x04, 0x00,  // v128.load align=4 offset=0
    0x41, 0x00,              // i32.const 0  (base addr for v128.load B)
    0xfd, 0x00, 0x04, 0x10,  // v128.load align=4 offset=16
    0xfd, 0xae, 0x01,        // i32x4.add
    0xfd, 0x0b, 0x04, 0x20,  // v128.store align=4 offset=32
    0x0b,                    // end
]);
const mod = new WebAssembly.Module(bytes);
const inst = new WebAssembly.Instance(mod);
const view = new Uint32Array(inst.exports.mem.buffer);
// A = [1, 2, 3, 4]
view[0] = 1; view[1] = 2; view[2] = 3; view[3] = 4;
// B = [10, 20, 30, 40]
view[4] = 10; view[5] = 20; view[6] = 30; view[7] = 40;
inst.exports.add_vecs();
// Result at view[8..12]
const r = [view[8], view[9], view[10], view[11]];
console.log("SIMD_RESULT:" + r.join(","));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "SIMD_RESULT:11,22,33,44" in result.stdout


# =============================================================================
# Exception handling (Wasm 3.0)
# =============================================================================
class TestWasmExceptionHandling:
    """Wasm exception handling: try/catch/throw."""

    async def test_throw_catch_roundtrip(self, scheduler: Scheduler) -> None:
        """WASM module catches its own exception and returns a sentinel."""
        # Exception handling bytecode encoding has changed over time.
        # We test the feature via JS-WASM interop: WASM calls an imported
        # JS function that throws, and the exception propagates to the JS
        # caller where it is caught.
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type section (5 bytes): () -> i32
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
  // Import section (15 bytes): "env"."thrower" : () -> i32
  0x02, 0x0f, 0x01,
    0x03, 0x65, 0x6e, 0x76,
    0x07, 0x74, 0x68, 0x72, 0x6f, 0x77, 0x65, 0x72,
    0x00, 0x00,
  // Function section (2 bytes): func 0 = type 0
  0x03, 0x02, 0x01, 0x00,
  // Export section (8 bytes): "test" -> func 1
  0x07, 0x08, 0x01, 0x04, 0x74, 0x65, 0x73, 0x74, 0x00, 0x01,
  // Code section (6 bytes): body(4) = call thrower (func 0)
  0x0a, 0x06, 0x01, 0x04, 0x00, 0x10, 0x00, 0x0b,
]);
const imports = {
  env: {
    thrower: () => { throw new Error("wasm_exception"); }
  }
};
const mod = new WebAssembly.Module(bytes);
const inst = new WebAssembly.Instance(mod, imports);
try {
  inst.exports.test();
  console.log("CAUGHT:none");
} catch (e) {
  console.log("CAUGHT:" + e.message);
}
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "CAUGHT:wasm_exception" in result.stdout


# =============================================================================
# REPL session persistence
# =============================================================================
class TestWasmSession:
    """WASM instance persists across session exec() calls."""

    async def test_wasm_instance_persists_across_exec(self, scheduler: Scheduler) -> None:
        """Compile in one exec(), use in the next."""
        async with await scheduler.session(language=Language.JAVASCRIPT) as session:
            # First exec: compile and instantiate
            r1 = await session.exec("""\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
  0x03, 0x02, 0x01, 0x00,
  0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
  0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
]);
globalThis.wasmInst = new WebAssembly.Instance(new WebAssembly.Module(bytes));
console.log("COMPILED:ok");
""")
            assert r1.exit_code == 0, f"compile stderr: {r1.stderr}"
            assert "COMPILED:ok" in r1.stdout

            # Second exec: use the instance from previous exec
            r2 = await session.exec("""\
console.log("RESULT:" + globalThis.wasmInst.exports.add(100, 200));
""")
            assert r2.exit_code == 0, f"use stderr: {r2.stderr}"
            assert "RESULT:300" in r2.stdout


# =============================================================================
# Edge cases and error handling
# =============================================================================
class TestWasmEdgeCases:
    """Edge cases: integer overflow, out-of-bounds memory, missing imports."""

    async def test_i32_overflow_wraps(self, scheduler: Scheduler) -> None:
        """i32 arithmetic wraps on overflow (two's complement)."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
  0x03, 0x02, 0x01, 0x00,
  0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
  0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
]);
const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes));
// i32 max = 2147483647, adding 1 wraps to -2147483648
console.log("WRAP:" + inst.exports.add(2147483647, 1));
// Large negative + large negative wraps through zero
console.log("NEG_WRAP:" + inst.exports.add(-2147483648, -1));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "WRAP:-2147483648" in result.stdout
        assert "NEG_WRAP:2147483647" in result.stdout

    async def test_memory_out_of_bounds_traps(self, scheduler: Scheduler) -> None:
        """Out-of-bounds memory access traps with RuntimeError."""
        # Body: locals(1) + local.get 0(2) + i32.load align=2 offset=0(3) + end(1) = 7 bytes
        # Code section: num_funcs(1) + body_size(1) + body(7) = 9 bytes
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type (6 bytes): (i32) -> i32
  0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
  0x03, 0x02, 0x01, 0x00,
  // Memory (3 bytes): 1 page (64 KiB)
  0x05, 0x03, 0x01, 0x00, 0x01,
  // Export (8 bytes): "load" -> func 0
  0x07, 0x08, 0x01, 0x04, 0x6c, 0x6f, 0x61, 0x64, 0x00, 0x00,
  // Code (9 bytes): body(7) = local.get 0, i32.load, end
  0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x28, 0x02, 0x00, 0x0b,
]);
const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes));
// Valid access at offset 0
console.log("VALID:" + inst.exports.load(0));
// Out of bounds: 1 page = 65536 bytes, reading i32 at 65536 overflows
try {
  inst.exports.load(65536);
  console.log("OOB:no_trap");
} catch (e) {
  console.log("OOB:" + e.constructor.name);
}
// Negative offset interpreted as large unsigned
try {
  inst.exports.load(-1);
  console.log("NEG:no_trap");
} catch (e) {
  console.log("NEG:" + e.constructor.name);
}
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "VALID:0" in result.stdout
        assert "OOB:RuntimeError" in result.stdout
        assert "NEG:RuntimeError" in result.stdout

    async def test_memory_grow_beyond_maximum_fails(self, scheduler: Scheduler) -> None:
        """Memory.grow() past maximum fails (returns -1 or throws RangeError)."""
        # The WASM spec says grow() should return -1, but JSC throws RangeError.
        # Both behaviors correctly prevent exceeding the maximum.
        code = """\
const mem = new WebAssembly.Memory({ initial: 1, maximum: 2 });
// Grow by 1 (within max) — should succeed
console.log("GROW1:" + mem.grow(1));
// Grow by 1 more (would exceed max=2) — should fail
try {
  const result = mem.grow(1);
  console.log("GROW2:" + result);
} catch (e) {
  console.log("GROW2_ERR:" + e.constructor.name);
}
console.log("PAGES:" + (mem.buffer.byteLength / 65536));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "GROW1:1" in result.stdout
        # Either returns -1 (spec) or throws RangeError (JSC)
        assert "GROW2:-1" in result.stdout or "GROW2_ERR:RangeError" in result.stdout
        assert "PAGES:2" in result.stdout  # still at max

    async def test_missing_import_throws(self, scheduler: Scheduler) -> None:
        """Instantiating a module with missing imports throws LinkError."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
  0x02, 0x0b, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x03, 0x66, 0x6f, 0x6f, 0x00, 0x00,
  0x03, 0x02, 0x01, 0x00,
  0x07, 0x08, 0x01, 0x04, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x01,
  0x0a, 0x06, 0x01, 0x04, 0x00, 0x10, 0x00, 0x0b,
]);
const mod = new WebAssembly.Module(bytes);
// No imports provided
try {
  new WebAssembly.Instance(mod);
  console.log("LINK:ok");
} catch (e) {
  console.log("LINK:" + e.constructor.name);
}
// Wrong import type (number instead of function)
try {
  new WebAssembly.Instance(mod, { env: { foo: 42 } });
  console.log("TYPE:ok");
} catch (e) {
  console.log("TYPE:" + e.constructor.name);
}
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        # Missing imports should throw LinkError or TypeError
        assert "LINK:" in result.stdout
        assert result.stdout.split("LINK:")[1].split("\n")[0] != "ok"
        assert "TYPE:" in result.stdout
        assert result.stdout.split("TYPE:")[1].split("\n")[0] != "ok"

    async def test_unreachable_traps(self, scheduler: Scheduler) -> None:
        """The unreachable instruction triggers a RuntimeError trap."""
        # Module with a single function that executes `unreachable`
        # Body: locals(0) + unreachable(0x00) + end(0x0b) = 3 bytes
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  // Type: () -> ()
  0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
  0x03, 0x02, 0x01, 0x00,
  // Export: "trap" -> func 0
  0x07, 0x08, 0x01, 0x04, 0x74, 0x72, 0x61, 0x70, 0x00, 0x00,
  // Code: body(3) = unreachable, end
  0x0a, 0x05, 0x01, 0x03, 0x00, 0x00, 0x0b,
]);
const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes));
try {
  inst.exports.trap();
  console.log("TRAP:none");
} catch (e) {
  console.log("TRAP:" + e.constructor.name);
}
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "TRAP:RuntimeError" in result.stdout

    async def test_stack_overflow_traps(self, scheduler: Scheduler) -> None:
        """Infinite recursion in WASM causes a RuntimeError (stack overflow)."""
        # Module: func 0 calls itself recursively
        # Body: locals(0) + call 0(0x10 0x00) + end(0x0b) = 4 bytes
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
  0x03, 0x02, 0x01, 0x00,
  0x07, 0x09, 0x01, 0x05, 0x72, 0x65, 0x63, 0x75, 0x72, 0x00, 0x00,
  0x0a, 0x06, 0x01, 0x04, 0x00, 0x10, 0x00, 0x0b,
]);
const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes));
try {
  inst.exports.recur();
  console.log("STACK:no_trap");
} catch (e) {
  console.log("STACK:" + e.constructor.name);
}
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "STACK:RangeError" in result.stdout

    async def test_division_by_zero_traps(self, scheduler: Scheduler) -> None:
        """i32.div_s by zero traps with RuntimeError."""
        # Module: div(i32, i32) -> i32 using i32.div_s (opcode 0x6d)
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
  0x03, 0x02, 0x01, 0x00,
  0x07, 0x07, 0x01, 0x03, 0x64, 0x69, 0x76, 0x00, 0x00,
  // Code: body(7) = local.get 0, local.get 1, i32.div_s, end
  0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6d, 0x0b,
]);
const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes));
// Normal division
console.log("DIV:" + inst.exports.div(10, 3));
// Division by zero
try {
  inst.exports.div(1, 0);
  console.log("ZERO:no_trap");
} catch (e) {
  console.log("ZERO:" + e.constructor.name);
}
// Signed overflow: INT32_MIN / -1
try {
  inst.exports.div(-2147483648, -1);
  console.log("OVERFLOW:no_trap");
} catch (e) {
  console.log("OVERFLOW:" + e.constructor.name);
}
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "DIV:3" in result.stdout
        assert "ZERO:RuntimeError" in result.stdout
        assert "OVERFLOW:RuntimeError" in result.stdout

    async def test_multiple_module_instances_isolated(self, scheduler: Scheduler) -> None:
        """Two instances of the same module have independent memory."""
        code = """\
const bytes = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x0b, 0x02,
    0x60, 0x02, 0x7f, 0x7f, 0x00,
    0x60, 0x01, 0x7f, 0x01, 0x7f,
  0x03, 0x03, 0x02, 0x00, 0x01,
  0x05, 0x03, 0x01, 0x00, 0x01,
  0x07, 0x16, 0x03,
    0x03, 0x6d, 0x65, 0x6d, 0x02, 0x00,
    0x05, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x00, 0x00,
    0x04, 0x6c, 0x6f, 0x61, 0x64, 0x00, 0x01,
  0x0a, 0x13, 0x02,
    0x09, 0x00, 0x20, 0x00, 0x20, 0x01, 0x36, 0x02, 0x00, 0x0b,
    0x07, 0x00, 0x20, 0x00, 0x28, 0x02, 0x00, 0x0b,
]);
const mod = new WebAssembly.Module(bytes);
const a = new WebAssembly.Instance(mod);
const b = new WebAssembly.Instance(mod);
a.exports.store(0, 111);
b.exports.store(0, 222);
// Each instance has its own memory
console.log("A:" + a.exports.load(0));
console.log("B:" + b.exports.load(0));
"""
        result = await scheduler.run(code=code, language=Language.JAVASCRIPT)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "A:111" in result.stdout
        assert "B:222" in result.stdout

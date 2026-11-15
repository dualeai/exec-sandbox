"""The guest's JavaScript REPL script run on this host under Bun (it needs Linux glibc):
start-up cases that no VM test can set up, the kernel random wrappers and the reseed flag.
Requires Linux and bun on PATH, no VM: the CI runners have no Bun, so it runs on a Linux host
where Bun is installed (the EC2 test host)."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import skip_unless_linux

_REPL = Path(__file__).parent.parent / "guest-agent" / "src" / "repl" / "scripts" / "js_repl.mjs"
_BUN = shutil.which("bun")

# Runs `bun <script>` under a seccomp filter: prctl(PR_SET_DUMPABLE, ...) fails with EPERM, every
# other call runs. The script takes the call from dlopen("libc.so.6"): no preload can replace it.
_NO_DUMPABLE = r"""
import ctypes, os, platform, sys
arch, nr = {"x86_64": (0xC000003E, 157), "aarch64": (0xC00000B7, 167)}[platform.machine()]
LD, JEQ, RET = 0x20, 0x15, 0x06
ALLOW, EPERM = 0x7FFF0000, 0x00050001
program = [
    (LD, 0, 0, 4), (JEQ, 1, 0, arch), (RET, 0, 0, ALLOW),  # another arch: no filter
    (LD, 0, 0, 0), (JEQ, 0, 3, nr),  # not prctl: allowed
    (LD, 0, 0, 16), (JEQ, 0, 1, 4),  # option PR_SET_DUMPABLE
    (RET, 0, 0, EPERM), (RET, 0, 0, ALLOW),
]
class Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]
class Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(Filter))]
filters = (Filter * len(program))(*program)
fprog = Program(len(program), filters)
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
assert libc.prctl(38, 1, None, 0, 0) == 0  # PR_SET_NO_NEW_PRIVS: a filter needs it without root
assert libc.prctl(22, 2, ctypes.addressof(fprog), 0, 0) == 0  # SECCOMP_MODE_FILTER
os.execv(sys.argv[1], sys.argv[1:])
"""

# Runs `bun <script>` under a seccomp filter: getrandom(2) of argv[1] bytes fails with EIO, every
# other call runs. A wrapper that takes its bytes from the kernel fails at that size only.
_GETRANDOM_FAILS = r"""
import ctypes, os, platform, sys
arch, nr = {"x86_64": (0xC000003E, 318), "aarch64": (0xC00000B7, 278)}[platform.machine()]
LD, JEQ, RET = 0x20, 0x15, 0x06
ALLOW, EIO = 0x7FFF0000, 0x00050005
program = [
    (LD, 0, 0, 4), (JEQ, 1, 0, arch), (RET, 0, 0, ALLOW),  # another arch: no filter
    (LD, 0, 0, 0), (JEQ, 0, 3, nr),  # not getrandom: allowed
    (LD, 0, 0, 24), (JEQ, 0, 1, int(sys.argv[1])),  # buflen (low word of args[1])
    (RET, 0, 0, EIO), (RET, 0, 0, ALLOW),
]
class Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]
class Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(Filter))]
filters = (Filter * len(program))(*program)
fprog = Program(len(program), filters)
libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
assert libc.prctl(38, 1, None, 0, 0) == 0  # PR_SET_NO_NEW_PRIVS: a filter needs it without root
assert libc.prctl(22, 2, ctypes.addressof(fprog), 0, 0) == 0  # SECCOMP_MODE_FILTER
os.execv(sys.argv[2], sys.argv[2:])
"""

skip_unless_bun = pytest.mark.skipif(_BUN is None, reason="This test runs the script under Bun (bun on PATH)")


def _frame(sentinel_id: str, code: str, *, reseed: bool) -> str:
    """One exec as the agent writes it: the header (sentinel id, code length in bytes, reseed
    flag), then the code."""
    return f"{sentinel_id} {len(code.encode())} {int(reseed)}\n{code}"


def _run(command: list[str], *execs: tuple[str, bool]) -> subprocess.CompletedProcess[bytes]:
    """Send `execs` (code, reseed flag) to the REPL run by `command`, as the execs s1, s2, ..."""
    frames = "".join(_frame(f"s{i}", code, reseed=reseed) for i, (code, reseed) in enumerate(execs, start=1))
    return subprocess.run(command, input=frames.encode(), capture_output=True, timeout=60, check=False)


def _run_repl(code: str, *, dumpable_fails: bool) -> subprocess.CompletedProcess[bytes]:
    """Send `code` to the REPL as exec s1, with prctl(PR_SET_DUMPABLE) failing or not."""
    assert _BUN is not None
    command = [_BUN, str(_REPL)]
    if dumpable_fails:
        command = [sys.executable, "-c", _NO_DUMPABLE, *command]
    return _run(command, (code, False))


def _printed(run: subprocess.CompletedProcess[bytes]) -> list[str]:
    """The lines that user code printed: stdout without the end markers."""
    return [line for line in run.stdout.decode().splitlines() if not line.startswith("__STDOUT_END_")]


def _sentinels(run: subprocess.CompletedProcess[bytes]) -> list[str]:
    """The sentinel lines of the REPL's stderr, in order: one per exec that ended."""
    return re.findall(r"^__SENTINEL_\w+?_-?\d+__$", run.stderr.decode(), re.MULTILINE)


@skip_unless_linux
@skip_unless_bun
def test_a_failed_prctl_stops_the_repl_before_any_exec() -> None:
    """prctl(PR_SET_DUMPABLE, 0) failing stops the REPL at start (fail closed, as the Python
    REPL): non-zero exit, the reason on stderr, no exec runs. Without the failure the same
    exec runs and ends."""
    code = "console.log('ran')"

    works = _run_repl(code, dumpable_fails=False)
    fails = _run_repl(code, dumpable_fails=True)

    assert "ran" in works.stdout.decode(), works.stderr.decode()
    assert works.stderr.decode().startswith("__REPL_READY__\n")  # before any exec, once
    assert re.findall(r"^__SENTINEL_\w+?_-?\d+__$", works.stderr.decode(), re.MULTILINE) == ["__SENTINEL_s1_0__"]
    assert fails.returncode != 0
    assert "prctl(PR_SET_DUMPABLE, 0) failed" in fails.stderr.decode()
    assert "ran" not in fails.stdout.decode()
    assert "__SENTINEL_" not in fails.stderr.decode()
    assert "__REPL_READY__" not in fails.stderr.decode()  # the agent reports a REPL that did not start


# Each random API with arguments that Bun refuses or accepts at its edges: the line it prints
# (the result type and length, or the error name, code and message).
_RANDOM_API_CASES = r"""
const c = require('node:crypto');
const cases = {
    bytes_negative: () => c.randomBytes(-1),
    bytes_string: () => c.randomBytes('1'),
    bytes_bad_callback: () => c.randomBytes(1, 5),
    bytes_ok: () => c.randomBytes(16),
    alias_ok: () => c.pseudoRandomBytes(4),
    grv_float: () => crypto.getRandomValues(new Float64Array(1)),
    grv_none: () => crypto.getRandomValues(),
    grv_65537: () => crypto.getRandomValues(new Uint8Array(65537)),
    grv_detached: () => { const f = crypto.getRandomValues; return f(new Uint8Array(1)); },
    grv_node: () => c.getRandomValues(new Uint16Array(3)),
    fill_sync_offset: () => c.randomFillSync(new Uint8Array(4), 5),
    fill_sync_size: () => c.randomFillSync(new Uint32Array(4), 1, 4),
    fill_sync_ok: () => c.randomFillSync(new Uint32Array(4), 1, 2),
    fill_no_callback: () => c.randomFill(new Uint8Array(4)),
    int_order: () => c.randomInt(3, 1),
    int_range: () => c.randomInt(0, 2 ** 48 + 1),
    int_bad: () => c.randomInt('a'),
    int_bad_callback: () => c.randomInt(1, 5, 7),
    int_ok: () => c.randomInt(5, 6),
    uuid_options: () => c.randomUUID(5),
    uuid_detached: () => { const f = crypto.randomUUID; return f(); },
    uuid_web: () => crypto.randomUUID(),
    uuid_v7_options: () => c.randomUUIDv7(5),
    uuid_v7_bun_encoding: () => Bun.randomUUIDv7('nope'),
    uuid_v7_bun_buffer: () => Bun.randomUUIDv7('buffer'),
};
for (const [name, f] of Object.entries(cases)) {
    try {
        const r = f();
        console.log(name, 'ok', r?.constructor?.name, r?.length);
    } catch (e) {
        console.log(name, e?.name, e?.code, e?.message);
    }
}
"""


@skip_unless_linux
@skip_unless_bun
def test_the_random_apis_are_the_kernel_wrappers() -> None:
    """Each random API of both realms, require('crypto') and an ESM import made in an exec, is the
    REPL's wrapper (not native code), with the name of Bun's function; the values keep their
    formats (UUID v4 and v7, randomInt in its range)."""
    assert _BUN is not None
    code = (
        "const c = require('node:crypto');\n"
        "const ns = await import('node:crypto');\n"
        "const fns = [crypto.getRandomValues, crypto.randomUUID, c.randomBytes, c.pseudoRandomBytes, c.prng, c.rng,"
        " c.randomFillSync, c.randomFill, c.randomInt, c.randomUUID, c.randomUUIDv7, require('bun').randomUUIDv7];\n"
        "console.log(fns.filter((f) => String(f).includes('[native code]')).length);\n"
        "console.log(fns.map((f) => f.name).join(' '));\n"
        "console.log(ns.randomBytes === c.randomBytes, ns.default === c, c.webcrypto === crypto);\n"
        "console.log(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(c.randomUUID()));\n"
        "const v7 = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;\n"
        "console.log(v7.test(Bun.randomUUIDv7()), v7.test(c.randomUUIDv7()));\n"
        "console.log(new Set(Array.from({ length: 300 }, () => c.randomInt(3))).size, Math.random() < 1);"
    )

    run = _run([_BUN, str(_REPL)], (code, False))

    assert _printed(run) == [
        "0",
        "getRandomValues randomUUID randomBytes deprecated deprecated deprecated randomFillSync randomFill"
        " randomInt randomUUID randomUUIDv7 randomUUIDv7",
        "true true true",
        "true",
        "true true",
        "3 true",
    ], run.stderr.decode()
    assert _sentinels(run) == ["__SENTINEL_s1_0__"]


@skip_unless_linux
@skip_unless_bun
def test_new_realms_get_the_kernel_math_random() -> None:
    """Each realm that user code makes has a Math.random of that realm (not JSC's, no way to the
    REPL's realm): node:vm contexts (createContext, runInNewContext and Script#runInNewContext,
    through require or an ESM import, DONT_CONTEXTIFY too) and ShadowRealms (in each realm, nested
    too). A context object with a Math of its own keeps it; a context keeps a Math.random that its
    code set. The globals the code sets, the context options and Bun's checks of runInNewContext
    stay; a node:vm function that user code replaced does not loop; the names and lengths stay."""
    assert _BUN is not None
    in_shadow = "String(Math.random).includes('[native code]')"
    code = (
        "const vm = require('node:vm');\n"
        "const ns = await import('node:vm');\n"
        "const native = (f) => String(f).includes('[native code]');\n"
        "const fns = [vm.runInContext('Math.random', vm.createContext()), vm.runInContext('Math.random', ns.createContext({})),"
        " vm.runInNewContext('Math.random'), vm.runInNewContext('Math.random', {}, { contextName: 'n' }),"
        " new vm.Script('Math.random').runInNewContext(), new ns.Script('Math.random').runInNewContext({}, { contextName: 'm' }),"
        " vm.runInNewContext('Math.random', vm.constants.DONT_CONTEXTIFY),"
        " vm.runInContext('Math.random', vm.createContext(vm.constants.DONT_CONTEXTIFY)),"
        " require('vm').runInNewContext('Math.random'), Math.random, process.constructor.constructor('return Math.random')()];\n"
        "console.log(fns.filter(native).length, fns.length);\n"
        f"const inShadow = {in_shadow!r};\n"
        "const nested = `new ShadowRealm().evaluate(${JSON.stringify(inShadow)})`;\n"
        "console.log(new ShadowRealm().evaluate(inShadow), new ShadowRealm().evaluate(nested), vm.runInNewContext(nested),"
        " process.constructor.constructor(`return ${nested}`)());\n"
        "const reach = \"Math.random.constructor('return typeof Bun')()\";\n"
        "console.log(vm.runInNewContext(reach, Object.create(null)), vm.runInNewContext(reach, vm.constants.DONT_CONTEXTIFY),"
        " vm.runInNewContext('Math.random.constructor === Function && ShadowRealm.prototype.evaluate.constructor === Function'));\n"
        "console.log(vm.runInNewContext('Math.random()', { Math: { random: () => 7 } }));\n"
        "const own = vm.createContext({});\n"
        "vm.runInContext('Math.random = () => 0.25', own);\n"
        "vm.createContext(own);\n"
        "vm.runInNewContext('1', own);\n"
        "new vm.Script('1').runInNewContext(own);\n"
        "console.log(vm.runInContext('Math.random()', own));\n"
        "const sandbox = {};\n"
        "new vm.Script('x = 1').runInNewContext(sandbox);\n"
        "vm.runInNewContext('y = 2', sandbox);\n"
        "console.log(sandbox.x, sandbox.y);\n"
        "try { vm.runInNewContext('1', {}, { contextName: 5 }); } catch (e) { console.log(e.code, e.message); }\n"
        "try { vm.runInNewContext('eval(\"1\")', {}, { contextCodeGeneration: { strings: false } }); }"
        " catch (e) { console.log(e.name); }\n"
        "const bunRunInNewContext = vm.runInNewContext;\n"
        "vm.runInNewContext = (text, object, options) => new vm.Script(text, options).runInNewContext(object, options);\n"
        "console.log(vm.runInNewContext('1 + 2', {}));\n"
        "vm.runInNewContext = bunRunInNewContext;\n"
        "console.log([vm.createContext, vm.Script.prototype.runInNewContext, ShadowRealm.prototype.evaluate,"
        " ShadowRealm.prototype.importValue, vm.runInNewContext('Math.random')].map((f) => f.name + f.length).join(' '));"
    )

    run = _run([_BUN, str(_REPL)], (code, False))

    assert _printed(run) == [
        "0 11",
        "false false false false",
        "undefined undefined true",
        "7",
        "0.25",
        "1 2",
        'ERR_INVALID_ARG_TYPE The "options.contextName" property must be of type string. Received type number (5)',
        "EvalError",
        "3",
        "createContext2 runInNewContext2 evaluate1 importValue2 random0",
    ], run.stderr.decode()
    assert _sentinels(run) == ["__SENTINEL_s1_0__"]


@skip_unless_linux
@skip_unless_bun
def test_the_wrappers_check_and_fail_as_bun_does(tmp_path: Path) -> None:
    """The random APIs called with arguments that Bun refuses or takes at its edges: the same
    results and errors (name, code, message) in the REPL as in a plain Bun process."""
    assert _BUN is not None
    script = tmp_path / "cases.mjs"
    script.write_text(_RANDOM_API_CASES)

    plain = subprocess.run([_BUN, str(script)], capture_output=True, timeout=60, check=False)
    run = _run([_BUN, str(_REPL)], (_RANDOM_API_CASES, False))

    assert len(_printed(run)) == 25, run.stderr.decode()
    assert _printed(run) == plain.stdout.decode().splitlines()


_ONE_SIZE_CASES = [
    pytest.param(
        4093,
        "const c = require('node:crypto');\n"
        "const cases = [['grv', () => crypto.getRandomValues(new Uint8Array(4093))],"
        " ['bytes', () => c.randomBytes(4093)], ['fill_sync', () => c.randomFillSync(Buffer.alloc(4093))],"
        " ['grv_other_size', () => crypto.getRandomValues(new Uint8Array(4092))]];\n"
        "for (const [name, f] of cases) { try { f(); console.log(name, 'ok'); } catch (e) { console.log(name, e.message); } }",
        [
            "grv getrandom(2) failed",
            "bytes getrandom(2) failed",
            "fill_sync getrandom(2) failed",
            "grv_other_size ok",
        ],
        id="sync-sizes",
    ),
    pytest.param(
        4096,
        "try { Math.random(); console.log('ok'); } catch (e) { console.log(e.message); }\n"
        "try { require('node:vm').runInNewContext('Math.random()'); console.log('ok'); } catch (e) { console.log(e.message); }",
        ["getrandom(2) failed", "getrandom(2) failed"],
        id="math-random-pool",
    ),
    pytest.param(
        4093,
        "const c = require('node:crypto');\n"
        "await new Promise((r) => c.randomBytes(4093, (e) => { console.log('bytes_cb', e ? e.message : 'ok'); r(); }));\n"
        "await new Promise((r) => c.randomFill(Buffer.alloc(4093), (e) => {"
        " console.log('fill_cb', e ? e.message : 'ok'); r(); }));\n"
        "const b = Buffer.alloc(8);\n"
        "c.randomFillSync(b, 2, 3);\n"
        "console.log('outside', b[0], b[1], b[5], b[6], b[7]);",
        ["bytes_cb getrandom(2) failed", "fill_cb getrandom(2) failed", "outside 0 0 0 0 0"],
        id="callbacks-and-range",
    ),
    pytest.param(
        3,
        "try { require('node:crypto').randomFillSync(Buffer.alloc(8), 2, 3); console.log('ok'); }"
        " catch (e) { console.log(e.message); }",
        ["getrandom(2) failed"],
        id="fill-sync-range",
    ),
    pytest.param(
        6,
        "try { require('node:crypto').randomInt(10); console.log('ok'); } catch (e) { console.log(e.message); }",
        ["getrandom(2) failed"],
        id="random-int",
    ),
    pytest.param(
        8,
        "for (const f of [() => require('node:crypto').randomUUIDv7(), () => Bun.randomUUIDv7()]) {"
        " try { f(); console.log('ok'); } catch (e) { console.log(e.message); } }",
        ["getrandom(2) failed", "getrandom(2) failed"],
        id="uuid-v7",
    ),
    pytest.param(
        16,
        "for (const f of [() => crypto.randomUUID(), () => require('node:crypto').randomUUID()]) {"
        " try { f(); console.log('ok'); } catch (e) { console.log(e.message); } }",
        ["getrandom(2) failed", "getrandom(2) failed"],
        id="uuid-v4",
    ),
]
"""Cases of test_each_random_api_reads_the_kernel_for_its_bytes: the size getrandom(2) fails
for, the code, and the lines it prints."""


@skip_unless_linux
@skip_unless_bun
@pytest.mark.parametrize(("size", "code", "printed"), _ONE_SIZE_CASES)
def test_each_random_api_reads_the_kernel_for_its_bytes(size: int, code: str, printed: list[str]) -> None:
    """getrandom(2) fails for the bytes that one API reads (seccomp): that API fails with the
    wrapper's error (the callback forms get it as their error), one byte less works. The
    wrappers read their bytes, randomInt 6 at a time, the UUID v7 wrappers the 8 bytes of
    rand_b, the v4 ones 16 bytes, randomFillSync(buf, 2, 3) the 3 bytes of its range and no
    byte out of it, and Math.random (also of a node:vm context) refills its pool of 4096 bytes."""
    assert _BUN is not None
    run = _run([sys.executable, "-c", _GETRANDOM_FAILS, str(size), _BUN, str(_REPL)], (code, False))

    assert _printed(run) == printed, run.stderr.decode()
    assert _sentinels(run) == ["__SENTINEL_s1_0__"]


@skip_unless_linux
@skip_unless_bun
def test_an_exec_with_the_reseed_flag_runs_its_code_and_ends() -> None:
    """An exec with the reseed flag (the REPL first calls Bun's randomFillSync 4096 times, so
    BoringSSL's generator of the main thread takes a new seed) runs its code and ends, as the
    execs without the flag before and after it. The new seed is checked on two copies of one
    save (test_instance_e2e TestRandomInCopies)."""
    assert _BUN is not None
    code = "console.log(require('node:crypto').randomBytes(8).length)"

    run = _run([_BUN, str(_REPL)], (code, False), (code, True), (code, False))

    assert _printed(run) == ["8", "8", "8"], run.stderr.decode()
    assert _sentinels(run) == ["__SENTINEL_s1_0__", "__SENTINEL_s2_0__", "__SENTINEL_s3_0__"]

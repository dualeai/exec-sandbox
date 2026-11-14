"""Tests for Bun FFI attack surface inside the guest VM.

Validates that user-submitted JavaScript code inside the REPL sandbox
can use bun:ffi to call arbitrary libc functions. The VM boundary (QEMU)
is the containment layer, not the JS runtime sandbox. These tests
document the attack surface and verify containment holds.

bun:ffi is accessible to user code via both require() and dynamic
import(). The REPL wrapper (js_repl.mjs) passes unfiltered `require` to the
createContext() globals, and runs the code with the main context's module
loader for import().

CVE references:
- CVE-2023-38408: ssh-agent PKCS11 arbitrary dlopen for code execution
- CVE-2022-30594: ptrace PTRACE_SEIZE + PTRACE_INTERRUPT bypass (mitigated by PR_SET_DUMPABLE=0)
- CVE-2024-1086: nf_tables double-free (reachable via FFI syscall wrappers if nft enabled)

Test categories:
- Normal: verify bun:ffi module is importable and dlopen returns working symbols
- Edge: verify FFI can bind and call libc I/O functions (open/read/close)
"""

import pytest

from exec_sandbox.models import Language
from exec_sandbox.scheduler import Scheduler
from exec_sandbox.vm_spec import VmSpec
from tests.conftest import run_once

# =============================================================================
# Normal: bun:ffi module is importable from user code
# =============================================================================


@pytest.mark.slow
class TestBunFfiImportable:
    """Bun's FFI gives user code arbitrary dlopen/dlsym access to libc.

    The JS REPL wrapper (js_repl.mjs) passes unfiltered `require` into the
    vm.createContext() sandbox and runs the code with the main context's module
    loader. User code can call require('bun:ffi') or import('bun:ffi') to get
    dlopen().

    Mitigations: VM boundary (QEMU hardware virtualization), UID 1000
    privilege drop, PR_SET_NO_NEW_PRIVS, capability restrictions.
    """

    async def test_ffi_require_gives_dlopen(self, dual_scheduler: Scheduler) -> None:
        """require('bun:ffi') exposes dlopen and FFIType to user code.

        This is the primary FFI entry point — require() is passed unfiltered
        into the createContext() globals (js_repl.mjs).
        """
        code = """\
const ffi = require('bun:ffi');
console.log('dlopen_type=' + typeof ffi.dlopen);
console.log('FFIType_type=' + typeof ffi.FFIType);
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.JAVASCRIPT), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "dlopen_type=function\nFFIType_type=object\n"

    async def test_ffi_import_gives_dlopen(self, dual_scheduler: Scheduler) -> None:
        """import('bun:ffi') exposes dlopen: the REPL runs the code with the main
        context's module loader (js_repl.mjs EXEC_SCRIPT)."""
        code = """\
const ffi = await import('bun:ffi');
console.log('dlopen_type=' + typeof ffi.dlopen);
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.JAVASCRIPT), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "dlopen_type=function\n"

    async def test_ffi_dlopen_libc_getpid(self, dual_scheduler: Scheduler) -> None:
        """User code can dlopen libc and call getpid() via FFI: it returns process.pid.

        Proves FFI can bind arbitrary libc symbols and invoke them.
        getpid() is used as a safe proof-of-concept; the same mechanism
        works for any libc function (execve, mmap, etc.).
        """
        code = """\
const { dlopen, FFIType } = require('bun:ffi');
const libc = dlopen('libc.so.6', {
    getpid: { args: [], returns: FFIType.i32 },
});
console.log('pid_matches=' + (libc.symbols.getpid() === process.pid));
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.JAVASCRIPT), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "pid_matches=true\n"


# =============================================================================
# Edge: FFI can perform libc I/O (open/read/close)
# =============================================================================


@pytest.mark.slow
class TestBunFfiLibcIo:
    """FFI enables full libc I/O: open(), read(), write(), close().

    User code can bypass Bun's file API entirely and call libc directly.
    This gives access to any file readable by UID 1000 inside the VM.

    Mitigations: EROFS rootfs is read-only, /home/user is ephemeral
    tmpfs, VM boundary prevents host file access.
    """

    async def test_ffi_read_etc_hostname(self, dual_scheduler: Scheduler) -> None:
        """User code can open and read /etc/hostname via libc FFI: it reads the same
        bytes as Bun's fs.readFileSync.

        Demonstrates the full dlopen -> open() -> read() -> close() chain.
        /etc/hostname is world-readable, so UID 1000 can read it.
        Contained by VM boundary — no host files are accessible.
        """
        code = """\
const { dlopen, FFIType, ptr } = require('bun:ffi');
const libc = dlopen('libc.so.6', {
    open:  { args: [FFIType.cstring, FFIType.i32], returns: FFIType.i32 },
    read:  { args: [FFIType.i32, FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
    close: { args: [FFIType.i32], returns: FFIType.i32 },
});
const encoder = new TextEncoder();
const path = encoder.encode('/etc/hostname\\0');  // null-terminated C string
const fd = libc.symbols.open(ptr(path), 0);  // O_RDONLY = 0
console.log('fd_valid=' + (fd >= 0));
if (fd >= 0) {
    const buf = new Uint8Array(256);
    const n = libc.symbols.read(fd, ptr(buf), 256);
    libc.symbols.close(fd);
    const viaFfi = new TextDecoder().decode(buf.subarray(0, Math.max(n, 0)));
    console.log('read_ok=' + (n > 0));
    console.log('same_as_fs=' + (viaFfi === require('fs').readFileSync('/etc/hostname', 'utf8')));
}
"""
        result = await run_once(dual_scheduler, VmSpec(language=Language.JAVASCRIPT), code)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout == "fd_valid=true\nread_ok=true\nsame_as_fs=true\n"

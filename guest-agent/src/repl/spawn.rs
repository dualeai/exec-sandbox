//! REPL process lifecycle: spawn and write wrapper scripts.

use tokio::process::Command;

use crate::constants::*;
use crate::repl::wrappers::*;
use crate::types::Language;

/// State for a persistent REPL process.
pub(crate) struct ReplState {
    pub(crate) child: tokio::process::Child,
    pub(crate) stdin: tokio::process::ChildStdin,
    pub(crate) stdout: tokio::process::ChildStdout,
    pub(crate) stderr: tokio::process::ChildStderr,
}

/// Write REPL wrapper script to SANDBOX_ROOT for the given language.
pub(crate) async fn write_repl_wrapper(
    language: Language,
) -> Result<(), Box<dyn std::error::Error>> {
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
/// Scripts run from SANDBOX_ROOT so package resolution works naturally.
pub(crate) async fn spawn_repl(
    language: Language,
) -> Result<ReplState, Box<dyn std::error::Error>> {
    write_repl_wrapper(language).await?;

    let mut cmd = match language {
        Language::Python => {
            let mut c = Command::new("python3");
            c.arg(format!("{SANDBOX_ROOT}/_repl.py"));
            c.env(
                "PYTHONPATH",
                format!("{SANDBOX_ROOT}/site-packages:{PYTHON_SITE_PACKAGES}"),
            );
            c.env("PYTHONINTMAXSTRDIGITS", "0");
            c.env("PYTHONSAFEPATH", "1");
            // Redirect .pyc writes to writable tmpfs. Stdlib uses pre-compiled .pyc
            // from the image; user-installed packages get cached here for the session.
            c.env("PYTHONPYCACHEPREFIX", "/home/user/.pycache");
            // Replace musl's slow default malloc with jemalloc (~30% Python startup speedup).
            // musl's allocator uses a global lock; Python calls malloc ~26k times during init.
            // Note: inherited by subprocesses (gcc, etc.) — harmless but visible.
            // See: https://developers.home-assistant.io/blog/2020/07/13/alpine-python/
            c.env("LD_PRELOAD", "/usr/lib/libjemalloc.so.2");
            c
        }
        Language::Javascript => {
            let mut c = Command::new("bun");
            c.arg("--smol");
            c.arg(format!("{SANDBOX_ROOT}/_repl.mjs"));
            c.env(
                "NODE_PATH",
                format!("{SANDBOX_ROOT}/node_modules:{NODE_MODULES_SYSTEM}"),
            );
            c.env("BUN_JSC_useFTLJIT", "0");
            // Minimize background thread CPU when idle. Without these, JSC's
            // concurrent GC/JIT threads and Bun's GC timer cause ~20-25% CPU
            // even when the main thread is blocked in libc.read().
            // See: oven-sh/bun#27365, oven-sh/bun#21081
            c.env("BUN_GC_TIMER_DISABLE", "1"); // Disable Bun's 1s GC repeating timer
            c.env("BUN_JSC_useConcurrentGC", "false"); // No background GC marker threads
            c.env("BUN_JSC_useConcurrentJIT", "false"); // No background JIT compilation
            // Reduce idle RSS. Tighter growth factor limits heap expansion;
            // earlier critical threshold triggers GC sooner; mimalloc purge
            // returns freed pages to the OS immediately (important for balloon).
            c.env("BUN_JSC_miniVMHeapGrowthFactor", "1.05"); // 5% growth (default 1.20)
            c.env("BUN_JSC_criticalGCMemoryThreshold", "0.50"); // Aggressive GC at 50% of RAM
            c.env("MIMALLOC_PURGE_DELAY", "0"); // Immediate page release to OS
            c
        }
        Language::Raw => {
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

    // Defense-in-depth: harden the REPL child process before exec.
    // We handle uid/gid/prctl ALL in pre_exec to control ordering.
    unsafe {
        cmd.pre_exec(|| {
            // 1. Block privilege escalation via execve
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // 2-3. Drop to non-root sandbox user
            if libc::setgid(SANDBOX_GID) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::setuid(SANDBOX_UID) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // 4. Prevent ptrace attach (fork→exec window only)
            if libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let mut child = cmd.spawn()?;

    // 5. RLIMIT_NPROC=1024 via prlimit64 from parent (root)
    {
        let nproc_limit = libc::rlimit {
            rlim_cur: 1024,
            rlim_max: 1024,
        };
        let pid = match child.id() {
            Some(id) => id as libc::pid_t,
            None => {
                eprintln!("WARNING: REPL child exited before prlimit64(RLIMIT_NPROC)");
                0
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
                eprintln!("WARNING: prlimit64(RLIMIT_NPROC) failed for pid {pid}: {err}");
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

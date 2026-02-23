//! REPL wrapper scripts loaded via `include_str!()`.
//!
//! Each language has a wrapper script that implements the sentinel protocol:
//!   Input:  "{sentinel_id} {code_len}\n" + code_bytes on stdin
//!   Output: stdout/stderr from user code, then __SENTINEL_{id}_{exit_code}__\n on stderr

pub(crate) const PYTHON_REPL_WRAPPER: &str = include_str!("scripts/python_repl.py");
pub(crate) const JS_REPL_WRAPPER: &str = include_str!("scripts/js_repl.mjs");
pub(crate) const SHELL_REPL_WRAPPER: &str = include_str!("scripts/shell_repl.sh");

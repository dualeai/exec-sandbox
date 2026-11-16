//! Input validation for code execution and environment variables.

use std::collections::HashMap;

use crate::constants::*;
use crate::error::CmdError;
use crate::types::Language;

/// Validate code and timeout parameters.
pub(crate) fn validate_code_params(code: &str, timeout: u64) -> Result<(), CmdError> {
    if code.trim().is_empty() {
        return Err(CmdError::code("Code cannot be empty"));
    }
    if code.contains('\0') {
        return Err(CmdError::code("Code cannot contain null bytes"));
    }
    if code.len() > MAX_CODE_SIZE_BYTES {
        return Err(CmdError::code(format!(
            "Code too large: {} bytes (max {} bytes)",
            code.len(),
            MAX_CODE_SIZE_BYTES
        )));
    }
    if timeout > MAX_TIMEOUT_SECONDS {
        return Err(CmdError::code(format!(
            "Timeout too large: {}s (max {}s)",
            timeout, MAX_TIMEOUT_SECONDS
        )));
    }
    Ok(())
}

/// Validate environment variables.
pub(crate) fn validate_env_vars(env_vars: &HashMap<String, String>) -> Result<(), CmdError> {
    if env_vars.len() > MAX_ENV_VARS {
        return Err(CmdError::env_var(format!(
            "Too many environment variables: {} (max {})",
            env_vars.len(),
            MAX_ENV_VARS
        )));
    }

    for (key, value) in env_vars {
        if BLOCKED_ENV_VARS.contains(&key.to_ascii_uppercase().as_str()) {
            return Err(CmdError::env_var(format!(
                "Blocked environment variable: '{}' (security risk)",
                key
            )));
        }
        if key.is_empty() || key.len() > MAX_ENV_VAR_NAME_LENGTH {
            return Err(CmdError::env_var(format!(
                "Invalid environment variable name length: {} (max {})",
                key.len(),
                MAX_ENV_VAR_NAME_LENGTH
            )));
        }
        if value.len() > MAX_ENV_VAR_VALUE_LENGTH {
            return Err(CmdError::env_var(format!(
                "Environment variable value too large: {} bytes (max {})",
                value.len(),
                MAX_ENV_VAR_VALUE_LENGTH
            )));
        }
        if key.chars().any(is_forbidden_control_char) {
            return Err(CmdError::env_var(format!(
                "Environment variable name '{}' contains forbidden control character",
                key
            )));
        }
        if value.chars().any(is_forbidden_control_char) {
            return Err(CmdError::env_var(format!(
                "Environment variable '{}' value contains forbidden control character",
                key
            )));
        }
    }

    Ok(())
}

/// Check for control characters that are forbidden in env var names/values.
/// Allows: tab (0x09), printable ASCII (0x20-0x7E), every char from U+0080 up.
/// Forbids: NUL, C0 controls (except tab), DEL (0x7F).
/// Same set as ENV_VAR_FORBIDDEN_CONTROL_CHARS in src/exec_sandbox/constants.py.
fn is_forbidden_control_char(c: char) -> bool {
    let code = c as u32;
    code < 0x09 || (0x0A..0x20).contains(&code) || code == 0x7F
}

/// The code the REPL runs: `code` (a leading UTF-8 BOM removed) after a prefix that
/// sets `env_vars` for `language`.
///
/// In RAW mode, a key that is not a valid shell name is skipped with a warning.
/// For RAW mode with shebang lines (`#!`), the user code is written to a temp
/// file so the kernel's `binfmt_script` handles interpreter dispatch.
pub(crate) fn prepend_env_vars(
    language: Language,
    code: &str,
    env_vars: &HashMap<String, String>,
) -> String {
    // Strip UTF-8 BOM (U+FEFF)
    let code = code.strip_prefix('\u{FEFF}').unwrap_or(code);

    let mut full_code = String::new();
    // The body of a single-quoted Python or JavaScript string literal
    let quoted = |s: &str| s.replace('\\', "\\\\").replace('\'', "\\'");

    if !env_vars.is_empty() {
        match language {
            Language::Python => {
                full_code.push_str("import os as __os__\n");
                for (key, value) in env_vars {
                    let (key, value) = (quoted(key), quoted(value));
                    full_code.push_str(&format!("__os__.environ['{key}']='{value}'\n"));
                }
                full_code.push_str("del __os__\n");
            }
            Language::Javascript => {
                for (key, value) in env_vars {
                    let (key, value) = (quoted(key), quoted(value));
                    full_code.push_str(&format!("process.env['{key}']='{value}';\n"));
                }
            }
            Language::Raw => {
                for (key, value) in env_vars {
                    if !key
                        .bytes()
                        .next()
                        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
                        || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                    {
                        log_warn!("Skipping invalid shell env var key: {:?}", key);
                        continue;
                    }
                    let escaped_val = value.replace('\'', "'\"'\"'");
                    full_code.push_str(&format!("export {key}='{escaped_val}'\n"));
                }
            }
        }
    }

    // RAW mode shebang: write to temp file for kernel-level interpretation.
    if language == Language::Raw && code.starts_with("#!") {
        let mut sentinel = String::from("_EXEC_SANDBOX_EOF_");
        while code.lines().any(|line| line == sentinel) {
            sentinel.push('X');
        }
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // -------------------------------------------------------------------------
    // validate_code_params
    // -------------------------------------------------------------------------

    /// The reply of refused code parameters: (message, error type); None when accepted.
    fn code_refusal(code: &str, timeout: u64) -> Option<(String, &'static str)> {
        match validate_code_params(code, timeout) {
            Ok(()) => None,
            Err(CmdError::Reply {
                message,
                error_type,
            }) => Some((message, error_type)),
            Err(CmdError::Fatal(e)) => panic!("expected a reply error, got {e}"),
        }
    }

    #[rstest]
    #[case::code("print('hello')", 30)]
    #[case::timeout_zero("x=1", 0)] // the default deadline
    #[case::timeout_of_300_seconds("x=1", 300)]
    fn test_code_params_are_accepted(#[case] code: &str, #[case] timeout: u64) {
        assert_eq!(code_refusal(code, timeout), None);
    }

    #[rstest]
    #[case::empty("", 30, "Code cannot be empty")]
    #[case::whitespace_only("   \n\t  ", 30, "Code cannot be empty")]
    #[case::newlines_only("\n\n\n", 30, "Code cannot be empty")]
    #[case::null_byte("print('hi')\0print('bye')", 30, "Code cannot contain null bytes")]
    #[case::timeout_over_300_seconds("x=1", 301, "Timeout too large: 301s (max 300s)")]
    fn test_code_params_are_refused(
        #[case] code: &str,
        #[case] timeout: u64,
        #[case] message: &str,
    ) {
        assert_eq!(
            code_refusal(code, timeout),
            Some((message.to_string(), "code_error"))
        );
    }

    #[test]
    fn test_code_of_1_mib_is_accepted_one_byte_more_is_refused() {
        assert_eq!(code_refusal(&"x".repeat(1_048_576), 30), None);
        assert_eq!(
            code_refusal(&"x".repeat(1_048_577), 30),
            Some((
                "Code too large: 1048577 bytes (max 1048576 bytes)".to_string(),
                "code_error"
            ))
        );
    }

    // -------------------------------------------------------------------------
    // validate_env_vars
    // -------------------------------------------------------------------------

    #[rstest]
    #[case::none(HashMap::new())]
    #[case::two(HashMap::from([("FOO".into(), "bar".into()), ("BAZ".into(), "qux".into())]))]
    #[case::one_hundred((0..100).map(|i| (format!("VAR_{i}"), "val".into())).collect())]
    #[case::name_of_256_bytes(HashMap::from([("A".repeat(256), "val".into())]))]
    #[case::value_of_4096_bytes(HashMap::from([("FOO".into(), "x".repeat(4096))]))]
    #[case::tab_in_value(HashMap::from([("FOO".into(), "a\tb".into())]))]
    fn test_env_vars_are_accepted(#[case] env: HashMap<String, String>) {
        assert!(validate_env_vars(&env).is_ok());
    }

    /// The reply of refused env vars: (message, error type).
    fn refusal_of(env: &HashMap<String, String>) -> (String, &'static str) {
        match validate_env_vars(env) {
            Err(CmdError::Reply {
                message,
                error_type,
            }) => (message, error_type),
            other => panic!("expected a reply error, got {other:?}"),
        }
    }

    /// The reply of a refused `name=value`: (message, error type).
    fn refusal(name: &str, value: &str) -> (String, &'static str) {
        refusal_of(&HashMap::from([(name.to_string(), value.to_string())]))
    }

    #[test]
    fn test_more_than_100_env_vars_are_refused() {
        let env = (0..101)
            .map(|i| (format!("VAR_{i}"), "val".into()))
            .collect();
        assert_eq!(
            refusal_of(&env),
            (
                "Too many environment variables: 101 (max 100)".to_string(),
                "env_var_error"
            )
        );
    }

    #[rstest]
    #[case::empty("", "val", "Invalid environment variable name length: 0 (max 256)")]
    #[case::name_of_257_bytes(
        &"A".repeat(257),
        "val",
        "Invalid environment variable name length: 257 (max 256)"
    )]
    #[case::value_of_4097_bytes(
        "FOO",
        &"x".repeat(4097),
        "Environment variable value too large: 4097 bytes (max 4096)"
    )]
    fn test_env_var_of_a_refused_size(
        #[case] name: &str,
        #[case] value: &str,
        #[case] message: &str,
    ) {
        assert_eq!(refusal(name, value), (message.to_string(), "env_var_error"));
    }

    #[rstest]
    #[case("LD_PRELOAD")]
    #[case("ld_preload")] // the check ignores case
    #[case("LD_LIBRARY_PATH")]
    #[case("GLIBC_TUNABLES")]
    #[case("NODE_OPTIONS")]
    #[case("PYTHONPATH")]
    #[case("PYTHONSTARTUP")]
    #[case("BASH_ENV")]
    #[case("IFS")]
    #[case("PATH")]
    fn test_blocked_env_var_is_refused(#[case] name: &str) {
        assert_eq!(
            refusal(name, "/tmp/evil"),
            (
                format!("Blocked environment variable: '{name}' (security risk)"),
                "env_var_error"
            )
        );
    }

    #[rstest]
    #[case::nul("val\0ue")]
    #[case::escape("\x1b[31mred\x1b[0m")]
    #[case::newline("line1\nline2")]
    #[case::carriage_return("start\roverwrite")]
    #[case::bell("ding\x07")]
    #[case::del("delete\x7f")]
    fn test_control_character_in_value_is_refused(#[case] value: &str) {
        assert_eq!(
            refusal("FOO", value),
            (
                "Environment variable 'FOO' value contains forbidden control character".to_string(),
                "env_var_error"
            )
        );
    }

    #[rstest]
    #[case::nul("FOO\0BAR")]
    #[case::start_of_heading("F\x01OO")]
    #[case::del("FOO\x7F")]
    fn test_control_character_in_name_is_refused(#[case] name: &str) {
        assert_eq!(
            refusal(name, "value"),
            (
                format!("Environment variable name '{name}' contains forbidden control character"),
                "env_var_error"
            )
        );
    }

    // -------------------------------------------------------------------------
    // prepend_env_vars
    // -------------------------------------------------------------------------

    #[test]
    fn test_prepend_strips_bom_python() {
        let result = prepend_env_vars(Language::Python, "\u{FEFF}print('hello')", &HashMap::new());
        assert_eq!(result, "print('hello')");
    }

    #[test]
    fn test_prepend_strips_bom_javascript() {
        let result = prepend_env_vars(
            Language::Javascript,
            "\u{FEFF}console.log('hello')",
            &HashMap::new(),
        );
        assert_eq!(result, "console.log('hello')");
    }

    #[test]
    fn test_prepend_strips_bom_raw() {
        let result = prepend_env_vars(Language::Raw, "\u{FEFF}echo hello", &HashMap::new());
        assert_eq!(result, "echo hello");
    }

    #[test]
    fn test_prepend_no_bom_unchanged() {
        let result = prepend_env_vars(Language::Python, "print('hello')", &HashMap::new());
        assert_eq!(result, "print('hello')");
    }

    // The whole text of a shebang exec: the code goes to a temporary file through a quoted
    // heredoc (no expansion of $VAR or `cmd`), the file runs, and its status is the exec's
    // (`( exit $_sec )`: a bare `exit` would end the REPL). No line starts with a space
    #[test]
    fn test_prepend_shebang_runs_the_code_from_a_temporary_file() {
        let result = prepend_env_vars(Language::Raw, "#!/bin/sh\necho $VAR `cmd`", &HashMap::new());
        assert_eq!(
            result,
            concat!(
                "_sf=$(mktemp /tmp/exec_XXXXXX) || { printf 'exec-sandbox: mktemp failed\\n' >&2; exit 126; }\n",
                "cat > \"$_sf\" <<'_EXEC_SANDBOX_EOF_'\n",
                "#!/bin/sh\n",
                "echo $VAR `cmd`\n",
                "_EXEC_SANDBOX_EOF_\n",
                "chmod +x \"$_sf\"\n",
                "\"$_sf\"\n",
                "_sec=$?\n",
                "rm -f \"$_sf\"\n",
                "( exit $_sec )\n",
            )
        );
    }

    #[test]
    fn test_prepend_strips_bom_shebang() {
        let code = "#!/bin/sh\necho hello";
        assert_eq!(
            prepend_env_vars(Language::Raw, &format!("\u{FEFF}{code}"), &HashMap::new()),
            prepend_env_vars(Language::Raw, code, &HashMap::new())
        );
    }

    #[test]
    fn test_prepend_shebang_with_env_vars_exports_them_first() {
        let env = HashMap::from([("FOO".to_string(), "bar".to_string())]);
        let code = "#!/bin/sh\necho $FOO";
        assert_eq!(
            prepend_env_vars(Language::Raw, code, &env),
            format!(
                "export FOO='bar'\n{}",
                prepend_env_vars(Language::Raw, code, &HashMap::new())
            )
        );
    }

    #[test]
    fn test_prepend_no_shebang_passthrough() {
        let result = prepend_env_vars(Language::Raw, "echo hello", &HashMap::new());
        assert_eq!(result, "echo hello");
    }

    #[test]
    fn test_prepend_no_shebang_with_env() {
        let env = HashMap::from([("X".to_string(), "1".to_string())]);
        let result = prepend_env_vars(Language::Raw, "echo $X", &env);
        assert_eq!(result, "export X='1'\necho $X");
    }

    // The heredoc's end word: one X more while a line of the code is that word (a word
    // inside a line is no end)
    #[rstest]
    #[case::inside_a_line("#!/bin/sh\necho _EXEC_SANDBOX_EOF_", "_EXEC_SANDBOX_EOF_")]
    #[case::a_line("#!/bin/sh\n_EXEC_SANDBOX_EOF_", "_EXEC_SANDBOX_EOF_X")]
    #[case::a_line_and_inside_a_line(
        "#!/bin/sh\n_EXEC_SANDBOX_EOF_\necho _EXEC_SANDBOX_EOF_X",
        "_EXEC_SANDBOX_EOF_X"
    )]
    #[case::two_lines(
        "#!/bin/sh\n_EXEC_SANDBOX_EOF_\n_EXEC_SANDBOX_EOF_X",
        "_EXEC_SANDBOX_EOF_XX"
    )]
    fn test_prepend_shebang_heredoc_word_is_no_line_of_the_code(
        #[case] code: &str,
        #[case] word: &str,
    ) {
        let result = prepend_env_vars(Language::Raw, code, &HashMap::new());
        let lines: Vec<&str> = result.lines().collect();
        let end = 2 + code.lines().count(); // the line after the code
        assert_eq!(
            (lines[1], lines[end]),
            (format!("cat > \"$_sf\" <<'{word}'").as_str(), word)
        );
    }

    #[test]
    fn test_prepend_shebang_non_raw_ignored() {
        let code = "#!/usr/bin/awk -f\nBEGIN{print}";
        let result = prepend_env_vars(Language::Python, code, &HashMap::new());
        assert_eq!(result, code);
    }

    // A key and a value that both hold ' and \ (both pass the env checks): the whole
    // code, so a key that is not quoted, a value quoted twice or quoted user code fails here
    #[test]
    fn test_prepend_python_quotes_key_and_value() {
        let env = HashMap::from([("a'b\\c".into(), "it's\\x".into())]);
        let result = prepend_env_vars(Language::Python, "print('x')", &env);
        assert_eq!(
            result,
            "import os as __os__\n__os__.environ['a\\'b\\\\c']='it\\'s\\\\x'\ndel __os__\nprint('x')"
        );
    }

    #[test]
    fn test_prepend_js_quotes_key_and_value() {
        let env = HashMap::from([("a'b\\c".into(), "it's\\x".into())]);
        let result = prepend_env_vars(Language::Javascript, "console.log('x')", &env);
        assert_eq!(
            result,
            "process.env['a\\'b\\\\c']='it\\'s\\\\x';\nconsole.log('x')"
        );
    }

    #[test]
    fn test_prepend_raw_single_quote_escape() {
        let env = HashMap::from([("KEY".into(), "it's".into())]);
        let result = prepend_env_vars(Language::Raw, "echo hi", &env);
        assert_eq!(result, "export KEY='it'\"'\"'s'\necho hi");
    }

    #[rstest]
    #[case::space("BAD KEY")]
    #[case::digit_first("1KEY")]
    #[case::dash("A-B")]
    fn test_prepend_raw_skips_a_key_that_is_no_shell_name(#[case] key: &str) {
        let env = HashMap::from([(key.to_string(), "val".to_string())]);
        assert_eq!(prepend_env_vars(Language::Raw, "echo hi", &env), "echo hi");
    }
}

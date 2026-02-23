//! Input validation for code execution and environment variables.

use std::collections::HashMap;

use crate::constants::*;
use crate::types::Language;

/// Validate execution parameters (code, timeout, env vars).
pub(crate) fn validate_execute_params(
    code: &str,
    timeout: u64,
    env_vars: &HashMap<String, String>,
) -> Result<(), String> {
    if code.trim().is_empty() {
        return Err("Code cannot be empty".to_string());
    }
    if code.contains('\0') {
        return Err("Code cannot contain null bytes".to_string());
    }
    if code.len() > MAX_CODE_SIZE_BYTES {
        return Err(format!(
            "Code too large: {} bytes (max {} bytes)",
            code.len(),
            MAX_CODE_SIZE_BYTES
        ));
    }
    if timeout > MAX_TIMEOUT_SECONDS {
        return Err(format!(
            "Timeout too large: {}s (max {}s)",
            timeout, MAX_TIMEOUT_SECONDS
        ));
    }
    if env_vars.len() > MAX_ENV_VARS {
        return Err(format!(
            "Too many environment variables: {} (max {})",
            env_vars.len(),
            MAX_ENV_VARS
        ));
    }

    for (key, value) in env_vars {
        if BLOCKED_ENV_VARS.contains(&key.to_ascii_uppercase().as_str()) {
            return Err(format!(
                "Blocked environment variable: '{}' (security risk)",
                key
            ));
        }
        if key.is_empty() || key.len() > MAX_ENV_VAR_NAME_LENGTH {
            return Err(format!(
                "Invalid environment variable name length: {} (max {})",
                key.len(),
                MAX_ENV_VAR_NAME_LENGTH
            ));
        }
        if value.len() > MAX_ENV_VAR_VALUE_LENGTH {
            return Err(format!(
                "Environment variable value too large: {} bytes (max {})",
                value.len(),
                MAX_ENV_VAR_VALUE_LENGTH
            ));
        }
        if key.chars().any(is_forbidden_control_char) {
            return Err(format!(
                "Environment variable name '{}' contains forbidden control character",
                key
            ));
        }
        if value.chars().any(is_forbidden_control_char) {
            return Err(format!(
                "Environment variable '{}' value contains forbidden control character",
                key
            ));
        }
    }

    Ok(())
}

/// Check for control characters that are forbidden in env var names/values.
/// Allows: tab (0x09), printable ASCII (0x20-0x7E), UTF-8 continuation (0x80+).
/// Forbids: NUL, C0 controls (except tab), DEL (0x7F).
fn is_forbidden_control_char(c: char) -> bool {
    let code = c as u32;
    code < 0x09 || (0x0A..0x20).contains(&code) || code == 0x7F
}

/// Build code prefix that sets environment variables for the given language.
///
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

    if !env_vars.is_empty() {
        match language {
            Language::Python => {
                full_code.push_str("import os as __os__\n");
                for (key, value) in env_vars {
                    let escaped_key = key.replace('\\', "\\\\").replace('\'', "\\'");
                    let escaped_val = value.replace('\\', "\\\\").replace('\'', "\\'");
                    full_code.push_str(&format!(
                        "__os__.environ['{escaped_key}']='{escaped_val}'\n"
                    ));
                }
                full_code.push_str("del __os__\n");
            }
            Language::Javascript => {
                for (key, value) in env_vars {
                    let escaped_key = key.replace('\\', "\\\\").replace('\'', "\\'");
                    let escaped_val = value.replace('\\', "\\\\").replace('\'', "\\'");
                    full_code.push_str(&format!("process.env['{escaped_key}']='{escaped_val}';\n"));
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
                        eprintln!("Skipping invalid shell env var key: {:?}", key);
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

    // -------------------------------------------------------------------------
    // validate_execute_params
    // -------------------------------------------------------------------------

    #[test]
    fn test_valid_code_no_env() {
        let env = HashMap::new();
        assert!(validate_execute_params("print('hello')", 30, &env).is_ok());
    }

    #[test]
    fn test_valid_code_with_env() {
        let env = HashMap::from([("FOO".into(), "bar".into()), ("BAZ".into(), "qux".into())]);
        assert!(validate_execute_params("print('hello')", 30, &env).is_ok());
    }

    #[test]
    fn test_valid_code_timeout_zero() {
        assert!(validate_execute_params("x=1", 0, &HashMap::new()).is_ok());
    }

    #[test]
    fn test_code_exactly_max() {
        let code = "x".repeat(MAX_CODE_SIZE_BYTES);
        assert!(validate_execute_params(&code, 30, &HashMap::new()).is_ok());
    }

    #[test]
    fn test_code_exceeds_max() {
        let code = "x".repeat(MAX_CODE_SIZE_BYTES + 1);
        assert!(validate_execute_params(&code, 30, &HashMap::new()).is_err());
    }

    #[test]
    fn test_timeout_exactly_max() {
        assert!(validate_execute_params("x=1", MAX_TIMEOUT_SECONDS, &HashMap::new()).is_ok());
    }

    #[test]
    fn test_timeout_exceeds_max() {
        assert!(validate_execute_params("x=1", MAX_TIMEOUT_SECONDS + 1, &HashMap::new()).is_err());
    }

    #[test]
    fn test_exactly_max_env_vars() {
        let env: HashMap<String, String> = (0..MAX_ENV_VARS)
            .map(|i| (format!("VAR_{i}"), "val".into()))
            .collect();
        assert!(validate_execute_params("x=1", 30, &env).is_ok());
    }

    #[test]
    fn test_too_many_env_vars() {
        let env: HashMap<String, String> = (0..MAX_ENV_VARS + 1)
            .map(|i| (format!("VAR_{i}"), "val".into()))
            .collect();
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_env_var_name_exactly_max() {
        let key = "A".repeat(MAX_ENV_VAR_NAME_LENGTH);
        let env = HashMap::from([(key, "val".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_ok());
    }

    #[test]
    fn test_env_var_name_exceeds_max() {
        let key = "A".repeat(MAX_ENV_VAR_NAME_LENGTH + 1);
        let env = HashMap::from([(key, "val".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_env_var_value_exactly_max() {
        let val = "x".repeat(MAX_ENV_VAR_VALUE_LENGTH);
        let env = HashMap::from([("FOO".into(), val)]);
        assert!(validate_execute_params("x=1", 30, &env).is_ok());
    }

    #[test]
    fn test_env_var_value_exceeds_max() {
        let val = "x".repeat(MAX_ENV_VAR_VALUE_LENGTH + 1);
        let env = HashMap::from([("FOO".into(), val)]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_tab_in_env_var_value_allowed() {
        let env = HashMap::from([("FOO".into(), "a\tb".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_ok());
    }

    #[test]
    fn test_null_byte_in_code() {
        assert!(validate_execute_params("print('hi')\0print('bye')", 30, &HashMap::new()).is_err());
    }

    #[test]
    fn test_whitespace_only_code() {
        assert!(validate_execute_params("   \n\t  ", 30, &HashMap::new()).is_err());
    }

    #[test]
    fn test_newlines_only_code() {
        assert!(validate_execute_params("\n\n\n", 30, &HashMap::new()).is_err());
    }

    #[test]
    fn test_blocked_env_ld_preload() {
        let env = HashMap::from([("LD_PRELOAD".into(), "/evil.so".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_blocked_env_ld_preload_lowercase() {
        let env = HashMap::from([("ld_preload".into(), "/evil.so".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_blocked_env_path() {
        let env = HashMap::from([("PATH".into(), "/evil".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_blocked_env_bash_env() {
        let env = HashMap::from([("BASH_ENV".into(), "/evil".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_blocked_env_node_options() {
        let env = HashMap::from([("NODE_OPTIONS".into(), "--require=evil".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_blocked_env_pythonpath() {
        let env = HashMap::from([("PYTHONPATH".into(), "/tmp/evil".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_blocked_env_ifs() {
        let env = HashMap::from([("IFS".into(), " ".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_null_byte_in_env_name() {
        let env = HashMap::from([("FOO\0BAR".into(), "val".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_control_char_in_env_name() {
        let env = HashMap::from([("FOO\x01".into(), "val".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_esc_in_env_value() {
        let env = HashMap::from([("FOO".into(), "val\x1B".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
    }

    #[test]
    fn test_del_in_env_name() {
        let env = HashMap::from([("FOO\x7F".into(), "val".into())]);
        assert!(validate_execute_params("x=1", 30, &env).is_err());
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
    fn test_prepend_strips_bom_raw() {
        let result = prepend_env_vars(Language::Raw, "\u{FEFF}echo hello", &HashMap::new());
        assert_eq!(result, "echo hello");
    }

    #[test]
    fn test_prepend_strips_bom_shebang() {
        let result = prepend_env_vars(
            Language::Raw,
            "\u{FEFF}#!/bin/sh\necho hello",
            &HashMap::new(),
        );
        assert!(result.contains("mktemp"));
    }

    #[test]
    fn test_prepend_no_bom_unchanged() {
        let result = prepend_env_vars(Language::Python, "print('hello')", &HashMap::new());
        assert_eq!(result, "print('hello')");
    }

    #[test]
    fn test_prepend_shebang_wraps_tempfile() {
        let code = "#!/usr/bin/awk -f\nBEGIN{print}";
        let result = prepend_env_vars(Language::Raw, code, &HashMap::new());
        assert!(result.contains("mktemp"));
        assert!(result.contains("cat >"));
        assert!(result.contains("chmod +x"));
        assert!(result.contains("#!/usr/bin/awk -f\nBEGIN{print}"));
    }

    #[test]
    fn test_prepend_shebang_with_env_vars() {
        let env = HashMap::from([("FOO".to_string(), "bar".to_string())]);
        let result = prepend_env_vars(Language::Raw, "#!/bin/sh\necho $FOO", &env);
        let export_pos = result.find("export FOO='bar'").unwrap();
        let mktemp_pos = result.find("mktemp").unwrap();
        assert!(export_pos < mktemp_pos);
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
        assert!(result.contains("export X='1'"));
        assert!(result.contains("echo $X"));
        assert!(!result.contains("mktemp"));
    }

    #[test]
    fn test_prepend_shebang_sentinel_no_collision_substring() {
        // Sentinel as a substring of a line should NOT trigger collision avoidance
        let code = "#!/bin/sh\necho _EXEC_SANDBOX_EOF_";
        let result = prepend_env_vars(Language::Raw, code, &HashMap::new());
        assert!(result.contains("<<'_EXEC_SANDBOX_EOF_'"));
    }

    #[test]
    fn test_prepend_shebang_sentinel_collision_own_line() {
        // Sentinel on its own line MUST trigger collision avoidance
        let code = "#!/bin/sh\n_EXEC_SANDBOX_EOF_";
        let result = prepend_env_vars(Language::Raw, code, &HashMap::new());
        assert!(result.contains("_EXEC_SANDBOX_EOF_X"));
    }

    #[test]
    fn test_prepend_shebang_sentinel_double_collision() {
        let code = "#!/bin/sh\n_EXEC_SANDBOX_EOF_\necho _EXEC_SANDBOX_EOF_X";
        let result = prepend_env_vars(Language::Raw, code, &HashMap::new());
        // First sentinel collides, appends X. Second sentinel is a substring, not own line.
        assert!(result.contains("<<'_EXEC_SANDBOX_EOF_X'"));
    }

    #[test]
    fn test_prepend_shebang_non_raw_ignored() {
        let code = "#!/usr/bin/awk -f\nBEGIN{print}";
        let result = prepend_env_vars(Language::Python, code, &HashMap::new());
        assert_eq!(result, code);
    }

    #[test]
    fn test_prepend_shebang_quoted_heredoc() {
        let code = "#!/bin/sh\necho $VAR `cmd`";
        let result = prepend_env_vars(Language::Raw, code, &HashMap::new());
        assert!(result.contains("<<'_EXEC_SANDBOX_EOF_'"));
    }

    #[test]
    fn test_prepend_shebang_no_leading_whitespace() {
        let code = "#!/bin/sh\necho hi";
        let result = prepend_env_vars(Language::Raw, code, &HashMap::new());
        for (i, line) in result.lines().enumerate() {
            assert!(
                !line.starts_with(' ') && !line.starts_with('\t'),
                "line {} has leading whitespace: {:?}",
                i,
                line,
            );
        }
    }

    #[test]
    fn test_prepend_shebang_mktemp_guard() {
        let code = "#!/bin/sh\necho hi";
        let result = prepend_env_vars(Language::Raw, code, &HashMap::new());
        assert!(result.contains("|| {") && result.contains("exit 126"));
    }

    #[test]
    fn test_prepend_python_single_quote_escape() {
        let env = HashMap::from([("KEY".into(), "it's".into())]);
        let result = prepend_env_vars(Language::Python, "x=1", &env);
        assert!(result.contains("it\\'s"));
    }

    #[test]
    fn test_prepend_python_backslash_escape() {
        let env = HashMap::from([("KEY".into(), "C:\\path".into())]);
        let result = prepend_env_vars(Language::Python, "x=1", &env);
        assert!(result.contains("C:\\\\path"));
    }

    #[test]
    fn test_prepend_js_single_quote_escape() {
        let env = HashMap::from([("KEY".into(), "it's".into())]);
        let result = prepend_env_vars(Language::Javascript, "x=1", &env);
        assert!(result.contains("it\\'s"));
    }

    #[test]
    fn test_prepend_raw_single_quote_escape() {
        let env = HashMap::from([("KEY".into(), "it's".into())]);
        let result = prepend_env_vars(Language::Raw, "echo hi", &env);
        assert!(result.contains("'\"'\"'"));
    }

    #[test]
    fn test_prepend_raw_skip_space_in_key() {
        let env = HashMap::from([("BAD KEY".into(), "val".into())]);
        let result = prepend_env_vars(Language::Raw, "echo hi", &env);
        assert!(!result.contains("export BAD KEY"));
    }

    #[test]
    fn test_prepend_raw_skip_digit_start_key() {
        let env = HashMap::from([("1KEY".into(), "val".into())]);
        let result = prepend_env_vars(Language::Raw, "echo hi", &env);
        assert!(!result.contains("export 1KEY"));
    }
}

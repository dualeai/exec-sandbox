//! Package installation (pip, bun) with validation.

use tokio::process::Command;

use crate::constants::*;
use crate::error::*;
use crate::types::{GuestResponse, Language};

/// Validate a package name + version specifier.
///
/// Replaces the `regex` crate dependency. Pattern:
///   name: `[a-zA-Z0-9_\-\.]+`
///   operator: `[@=<>~]` (at least one required)
///   version: `[a-zA-Z0-9_\-\.@=<>~\^\*\[\], ]*` (no '/' — prevents local-path injection)
pub(crate) fn validate_package_name(pkg: &str) -> Result<(), String> {
    if pkg.is_empty() {
        return Err("Package name cannot be empty".into());
    }
    if pkg.len() > MAX_PACKAGE_NAME_LENGTH {
        return Err(format!(
            "Package name too long: {} bytes (max {})",
            pkg.len(),
            MAX_PACKAGE_NAME_LENGTH
        ));
    }
    if pkg.contains("..") || pkg.contains('/') || pkg.contains('\\') {
        return Err(format!(
            "Invalid package name: '{}' (path characters not allowed)",
            pkg
        ));
    }
    if pkg.contains('\0') {
        return Err("Package name contains null byte".into());
    }
    if pkg.chars().any(|c| c.is_control()) {
        return Err(format!(
            "Invalid package name: '{}' (control characters not allowed)",
            pkg
        ));
    }

    // Find the first version operator character
    let name_end = pkg.find(['@', '=', '<', '>', '~']).ok_or_else(|| {
        format!(
            "Invalid package name: '{}' (contains invalid characters)",
            pkg
        )
    })?;

    if name_end == 0 {
        return Err(format!(
            "Invalid package name: '{}' (contains invalid characters)",
            pkg
        ));
    }

    let name = &pkg[..name_end];
    let version_part = &pkg[name_end..];

    // Validate name: [a-zA-Z0-9_\-\.]+
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    {
        return Err(format!(
            "Invalid package name: '{}' (contains invalid characters)",
            pkg
        ));
    }

    // Require non-empty version after operator (reject e.g. "numpy=")
    if version_part.len() <= 1
        || version_part[1..]
            .bytes()
            .all(|b| matches!(b, b'@' | b'=' | b'<' | b'>' | b'~'))
    {
        return Err(format!(
            "Invalid package specifier: '{}' (empty version after operator)",
            pkg
        ));
    }

    // Validate version part: [a-zA-Z0-9_\-\.@=<>~\^\*\[\], ]*
    // Note: '/' is intentionally excluded to prevent local-path injection (e.g. "a@/etc/passwd")
    if !version_part.bytes().all(|b| {
        b.is_ascii_alphanumeric()
            || matches!(
                b,
                b'_' | b'-'
                    | b'.'
                    | b'@'
                    | b'='
                    | b'<'
                    | b'>'
                    | b'~'
                    | b'^'
                    | b'*'
                    | b'['
                    | b']'
                    | b','
                    | b' '
            )
    }) {
        return Err(format!(
            "Invalid package name: '{}' (contains invalid characters)",
            pkg
        ));
    }

    Ok(())
}

/// Install packages to system paths for snapshot persistence.
pub(crate) async fn install_packages(
    language: Language,
    packages: &[String],
    writer: &ResponseWriter,
) -> Result<(), CmdError> {
    use std::time::Instant;
    use tokio::time::Duration;

    if language == Language::Raw {
        return Err(CmdError::validation(format!(
            "Unsupported language '{}' for package installation (supported: python, javascript)",
            language.as_str()
        )));
    }
    if packages.is_empty() {
        return Err(CmdError::validation(
            "No packages specified for installation",
        ));
    }
    if packages.len() > MAX_PACKAGES {
        return Err(CmdError::validation(format!(
            "Too many packages: {} (max {})",
            packages.len(),
            MAX_PACKAGES
        )));
    }
    for pkg in packages {
        validate_package_name(pkg).map_err(CmdError::validation)?;
    }

    let start = Instant::now();

    let mut cmd = match language {
        Language::Python => {
            let mut c = Command::new("uv");
            c.arg("pip")
                .arg("install")
                .arg("--python")
                .arg("/opt/python/bin/python3")
                .arg("--target")
                .arg(PYTHON_SITE_PACKAGES);
            for pkg in packages {
                c.arg(pkg);
            }
            c.current_dir(PYTHON_SITE_PACKAGES);
            c.process_group(0);
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            c
        }
        Language::Javascript => {
            let mut c = Command::new("bun");
            c.arg("add");
            for pkg in packages {
                c.arg(pkg);
            }
            c.current_dir("/usr/local/lib");
            c.process_group(0);
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            c
        }
        Language::Raw => unreachable!(),
    };

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return Err(CmdError::execution(format!(
                "Failed to execute package manager for {}: {}",
                language.as_str(),
                e
            )));
        }
    };

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let stdout_task = spawn_output_reader(stdout, MAX_PACKAGE_OUTPUT_BYTES);
    let stderr_task = spawn_output_reader(stderr, MAX_PACKAGE_OUTPUT_BYTES);

    let wait_result = tokio::time::timeout(
        Duration::from_secs(PACKAGE_INSTALL_TIMEOUT_SECONDS),
        child.wait(),
    )
    .await;

    let status = match wait_result {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            let _ = graceful_terminate_process_group(&mut child, TERM_GRACE_PERIOD_SECONDS).await;
            return Err(CmdError::execution(format!("Process wait error: {}", e)));
        }
        Err(_) => {
            let _ = graceful_terminate_process_group(&mut child, TERM_GRACE_PERIOD_SECONDS).await;
            return Err(CmdError::timeout(format!(
                "Package installation timeout after {}s",
                PACKAGE_INSTALL_TIMEOUT_SECONDS
            )));
        }
    };

    let stdout_lines = stdout_task.await.unwrap_or_default();
    let stderr_lines = stderr_task.await.unwrap_or_default();

    let duration_ms = start.elapsed().as_millis() as u64;
    let exit_code = exit_code_from_status(status);

    // Sync filesystem for snapshot safety
    if exit_code == 0 {
        unsafe { libc::sync() };
    }

    if !stdout_lines.is_empty() {
        writer
            .send(&GuestResponse::Stdout {
                chunk: stdout_lines.join("\n") + "\n",
            })
            .await?;
    }

    if !stderr_lines.is_empty() {
        writer
            .send(&GuestResponse::Stderr {
                chunk: stderr_lines.join("\n") + "\n",
            })
            .await?;
    }

    writer
        .send(&GuestResponse::Complete {
            exit_code,
            execution_time_ms: duration_ms,
            spawn_ms: None,
            process_ms: None,
        })
        .await?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("pandas==2.0.0", true)]
    #[case("lodash@4.17.21", true)]
    #[case("numpy>=1.0,<2.0", true)]
    #[case("a==1", true)]
    #[case("pkg~=2.0", true)]
    #[case("lodash@~4.17", true)]
    #[case("lodash@^4.0.0", true)]
    fn test_validate_package_name_valid(#[case] name: &str, #[case] valid: bool) {
        assert_eq!(validate_package_name(name).is_ok(), valid);
    }

    #[rstest]
    #[case("", false)]
    #[case("pandas", false)] // no version specifier
    #[case("../evil==1.0", false)] // path traversal
    #[case("pkg/../../etc==1.0", false)] // slash in name
    #[case("pkg\x00==1.0", false)] // null byte
    #[case("pkg\x01==1.0", false)] // control char
    #[case("pkg\\path==1.0", false)] // backslash
    #[case("numpy=", false)] // empty version after operator
    #[case("numpy==", false)] // only operators, no version
    #[case("a@/etc/passwd", false)] // local-path injection via slash in version
    fn test_validate_package_name_invalid(#[case] name: &str, #[case] valid: bool) {
        assert_eq!(validate_package_name(name).is_ok(), valid);
    }

    #[test]
    fn test_validate_package_name_exactly_max_length() {
        let name = format!("{}==1.0", "a".repeat(MAX_PACKAGE_NAME_LENGTH - 5));
        // "==1.0" is 5 chars, so total = MAX_PACKAGE_NAME_LENGTH
        assert!(validate_package_name(&name).is_ok());
    }

    #[test]
    fn test_validate_package_name_exceeds_max_length() {
        let name = format!("{}==1.0", "a".repeat(MAX_PACKAGE_NAME_LENGTH));
        assert!(validate_package_name(&name).is_err());
    }
}

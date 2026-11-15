//! Package installation (uv pip for Python, bun for JavaScript) with validation.

use tokio::process::Command;

use crate::constants::*;
use crate::error::*;
use crate::types::{GuestResponse, Language};

/// Validate a package name + version specifier.
///
/// Hand-written check (no `regex` crate) for this pattern:
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
    // The character sets below refuse '/', '\\', NUL and control characters;
    // ".." is made of allowed characters
    if pkg.contains("..") {
        return Err(format!(
            "Invalid package name: '{}' (path characters not allowed)",
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

/// The output of an installer that ended. Its process group is killed first: a daemon that a
/// postinstall script starts would keep the output pipes open, and the snapshot taken after the
/// install must hold no process. Then both readers are awaited until `deadline`, or
/// INSTALLER_OUTPUT_GRACE_MS from now when that is later (an installer that ended just before
/// its deadline). None: a process outside the group still holds a pipe then (both readers
/// are aborted).
async fn finish_installer_output(
    installer: &crate::init::WaitedChild,
    stdout: tokio::task::JoinHandle<Vec<String>>,
    stderr: tokio::task::JoinHandle<Vec<String>>,
    deadline: tokio::time::Instant,
) -> Option<(Vec<String>, Vec<String>)> {
    let _ = installer.signal_group(libc::SIGKILL); // ESRCH: no member left
    let (stdout_abort, stderr_abort) = (stdout.abort_handle(), stderr.abort_handle());
    let grace =
        tokio::time::Instant::now() + std::time::Duration::from_millis(INSTALLER_OUTPUT_GRACE_MS);
    match tokio::time::timeout_at(deadline.max(grace), async { (stdout.await, stderr.await) }).await
    {
        Ok((out, err)) => Some((out.unwrap_or_default(), err.unwrap_or_default())),
        Err(_) => {
            stdout_abort.abort();
            stderr_abort.abort();
            None
        }
    }
}

/// Install packages to system paths for snapshot persistence.
///
/// Runs as root: `uv pip install --target PYTHON_SITE_PACKAGES` for Python,
/// `bun add` in /usr/local/lib for JavaScript. Raw is refused. After the
/// process exits, sends the collected stdout and stderr (each capped at
/// MAX_PACKAGE_OUTPUT_BYTES) as one chunk each, then a Complete message.
pub(crate) async fn install_packages(
    language: Language,
    packages: &[String],
    timeout: u64,
    writer: &ResponseWriter,
) -> Result<(), CmdError> {
    use std::time::Instant;
    use tokio::time::Duration;

    if language == Language::Raw {
        return Err(CmdError::package(format!(
            "Unsupported language '{}' for package installation (supported: python, javascript)",
            language.as_str()
        )));
    }
    if packages.is_empty() {
        return Err(CmdError::package("No packages specified for installation"));
    }
    if packages.len() > MAX_PACKAGES {
        return Err(CmdError::package(format!(
            "Too many packages: {} (max {})",
            packages.len(),
            MAX_PACKAGES
        )));
    }
    for pkg in packages {
        validate_package_name(pkg).map_err(CmdError::package)?;
    }

    let start = Instant::now();
    crate::oom_guard::ensure_vm_usable().map_err(|e| CmdError::execution(e.to_string()))?;

    let mut cmd = match language {
        Language::Python => {
            let mut c = Command::new("uv");
            c.arg("pip")
                .arg("install")
                .arg("--no-build")
                // The package layer is read-only at run time: compile now, or every
                // import recompiles in memory.
                .arg("--compile-bytecode")
                .arg("--python")
                .arg(format!("{PYTHON_HOME}/bin/python3"))
                .arg("--target")
                .arg(PYTHON_SITE_PACKAGES)
                .current_dir(PYTHON_SITE_PACKAGES);
            c
        }
        Language::Javascript => {
            let mut c = Command::new("bun");
            c.arg("add").current_dir("/usr/local/lib");
            c
        }
        Language::Raw => unreachable!(),
    };
    cmd.args(packages)
        .process_group(0)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

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
    // No await since the spawn: the PID 1 reaper cannot have collected it yet.
    let mut child = crate::init::WaitedChild::register(child).map_err(|e| {
        CmdError::execution(format!(
            "Failed to watch the package manager for {}: {e}",
            language.as_str()
        ))
    })?;

    let stdout_task = spawn_output_reader(stdout, MAX_PACKAGE_OUTPUT_BYTES);
    let stderr_task = spawn_output_reader(stderr, MAX_PACKAGE_OUTPUT_BYTES);

    let timeout = effective_timeout_secs(timeout); // 0: the default deadline, as for an exec
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout);
    let wait_result = tokio::time::timeout_at(deadline, child.wait()).await;

    let status = match wait_result {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            graceful_terminate_process_group(&mut child, TERM_GRACE_PERIOD_SECONDS).await;
            stdout_task.abort();
            stderr_task.abort();
            return Err(CmdError::execution(format!("Process wait error: {}", e)));
        }
        Err(_) => {
            graceful_terminate_process_group(&mut child, TERM_GRACE_PERIOD_SECONDS).await;
            stdout_task.abort();
            stderr_task.abort();
            return Err(CmdError::timeout(format!(
                "Package installation timeout after {}s",
                timeout
            )));
        }
    };
    let observed_exit_code = exit_code_from_status(status);
    let exit_code = crate::oom_guard::reconcile_retirement_exit(observed_exit_code);
    let Some((stdout_lines, stderr_lines)) =
        finish_installer_output(&child, stdout_task, stderr_task, deadline).await
    else {
        return Err(CmdError::timeout(format!(
            "Package installation timeout after {}s: a process the installer started outside its group kept its output open",
            timeout
        )));
    };

    let duration_ms = start.elapsed().as_millis() as u64;
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

    /// An installer (`script` in sh, its own group) that has ended.
    async fn ended_installer(
        script: &str,
    ) -> (
        crate::init::WaitedChild,
        tokio::task::JoinHandle<Vec<String>>,
        tokio::task::JoinHandle<Vec<String>>,
    ) {
        let mut spawned = Command::new("sh")
            .args(["-c", script])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap();
        let stdout = spawn_output_reader(spawned.stdout.take().unwrap(), 1024);
        let stderr = spawn_output_reader(spawned.stderr.take().unwrap(), 1024);
        let mut child = crate::init::WaitedChild::register(spawned).unwrap();
        assert!(child.wait().await.unwrap().success());
        (child, stdout, stderr)
    }

    #[tokio::test]
    async fn installer_output_ends_with_a_daemon_left_in_its_group() {
        // The daemon holds both pipes: only the group SIGKILL ends them before the deadline
        let (installer, stdout, stderr) = ended_installer("sleep 30 & echo installed").await;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

        let started = std::time::Instant::now();
        let output = finish_installer_output(&installer, stdout, stderr, deadline).await;

        assert_eq!(output, Some((vec!["installed".to_string()], vec![])));
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    #[tokio::test]
    async fn installer_output_at_its_deadline_still_waits_for_the_pipes_to_close() {
        // The installer ended just before its deadline (passed here); a process outside its
        // group holds the pipes for 0.6 s more: the output comes, not the "kept open" error
        let (installer, stdout, stderr) = ended_installer(
            r#"f=$(mktemp -u); mkfifo "$f"; setsid sh -c 'echo > "$1"; exec sleep 0.6' sh "$f" & read -r _ < "$f"; rm -f "$f"; echo installed"#,
        )
        .await;
        let deadline = tokio::time::Instant::now();

        let output = finish_installer_output(&installer, stdout, stderr, deadline).await;

        assert_eq!(output, Some((vec!["installed".to_string()], vec![])));
    }

    #[tokio::test]
    async fn installer_output_waits_until_a_deadline_later_than_the_grace() {
        // A process outside the group holds the pipes for 1.5 s, past the 1 s grace but
        // before the install deadline: the readers wait for the deadline, not the grace
        let (installer, stdout, stderr) = ended_installer(
            r#"f=$(mktemp -u); mkfifo "$f"; setsid sh -c 'echo > "$1"; exec sleep 1.5' sh "$f" & read -r _ < "$f"; rm -f "$f"; echo installed"#,
        )
        .await;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(3);

        let output = finish_installer_output(&installer, stdout, stderr, deadline).await;

        assert_eq!(output, Some((vec!["installed".to_string()], vec![])));
    }

    #[tokio::test]
    async fn installer_output_stops_at_the_deadline_with_a_daemon_outside_its_group() {
        // setsid: the daemon leaves the group, so the SIGKILL misses it and it keeps the pipes.
        // The installer ends only once the daemon is in its own session (the FIFO).
        let (installer, stdout, stderr) = ended_installer(
            r#"f=$(mktemp -u); mkfifo "$f"; setsid sh -c 'echo > "$1"; exec sleep 30' sh "$f" & read -r _ < "$f"; rm -f "$f"; echo installed"#,
        )
        .await;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(200);

        let output = finish_installer_output(&installer, stdout, stderr, deadline).await;

        assert_eq!(output, None);
    }

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

    // Specs that pass every earlier check: only the name's or the version's character set
    // refuses them
    #[rstest]
    #[case("pa$ndas==1")] // a name character outside [a-zA-Z0-9_.-]
    #[case("pandas==1.0;id")] // a version character outside its set (a shell separator)
    #[case("pandas==1.0|x")]
    fn test_validate_package_name_character_sets(#[case] spec: &str) {
        assert_eq!(
            validate_package_name(spec).unwrap_err(),
            format!("Invalid package name: '{spec}' (contains invalid characters)")
        );
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

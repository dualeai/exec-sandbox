//! Error types and response helpers for the guest agent.

use tokio::sync::mpsc;

use crate::constants::VERSION;
use crate::types::{GuestResponse, Language};

// ============================================================================
// CmdError — the unified error type for command handlers
// ============================================================================

/// Unified error type for command handlers.
///
/// Two variants mirror the h2 crate's stream-error vs connection-error pattern:
/// - `Reply`: non-fatal — send error response to host, continue with next command.
/// - `Fatal`: connection-level — write queue closed or port I/O failure, drop connection.
pub(crate) enum CmdError {
    Reply {
        message: String,
        error_type: &'static str,
        op_id: Option<String>,
    },
    Fatal(Box<dyn std::error::Error + Send + Sync>),
}

impl CmdError {
    /// Validation error (no op_id).
    pub(crate) fn validation(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: "validation_error",
            op_id: None,
        }
    }

    /// I/O error with optional op_id.
    pub(crate) fn io(msg: impl Into<String>, op_id: Option<&str>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: "io_error",
            op_id: op_id.map(String::from),
        }
    }

    /// Timeout error (no op_id).
    pub(crate) fn timeout(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: "timeout_error",
            op_id: None,
        }
    }

    /// Execution error (no op_id).
    pub(crate) fn execution(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: "execution_error",
            op_id: None,
        }
    }

    /// Attach an op_id to a Reply error. No-op on Fatal.
    pub(crate) fn with_op_id(self, id: &str) -> Self {
        match self {
            Self::Reply {
                message,
                error_type,
                ..
            } => Self::Reply {
                message,
                error_type,
                op_id: Some(id.to_string()),
            },
            fatal => fatal,
        }
    }
}

impl std::fmt::Display for CmdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reply { message, .. } => write!(f, "{message}"),
            Self::Fatal(e) => write!(f, "fatal: {e}"),
        }
    }
}

impl From<mpsc::error::SendError<Vec<u8>>> for CmdError {
    fn from(_: mpsc::error::SendError<Vec<u8>>) -> Self {
        Self::Fatal("Write queue closed".into())
    }
}

// ============================================================================
// Response helpers
// ============================================================================

/// Serialize a GuestResponse to JSON + newline and queue it for sending.
pub(crate) async fn send_response(
    write_tx: &mpsc::Sender<Vec<u8>>,
    msg: &GuestResponse,
) -> Result<(), CmdError> {
    let mut bytes = serde_json::to_vec(msg).map_err(|e| CmdError::Fatal(e.into()))?;
    bytes.push(b'\n');
    write_tx.send(bytes).await?;
    Ok(())
}

/// Send an error response to the host.
pub(crate) async fn send_streaming_error(
    write_tx: &mpsc::Sender<Vec<u8>>,
    message: String,
    error_type: &str,
    op_id: Option<&str>,
) -> Result<(), CmdError> {
    send_response(
        write_tx,
        &GuestResponse::Error {
            message,
            error_type: error_type.to_string(),
            op_id: op_id.map(|s| s.to_string()),
            version: Some(VERSION),
        },
    )
    .await
}

// ============================================================================
// Shared utilities
// ============================================================================

/// Extract exit code following Unix shell conventions.
///
/// - Normal exit: returns the exit code (0-255)
/// - Signal kill: returns 128 + signal_number (e.g., SIGKILL → 137)
/// - Neither: returns 255 as "unknown error" fallback
pub(crate) fn exit_code_from_status(status: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .or_else(|| status.signal().map(|sig| 128 + sig))
        .unwrap_or(255)
}

/// Parse a language string, returning CmdError::Reply on failure.
pub(crate) fn parse_language(language_str: &str, context: &str) -> Result<Language, CmdError> {
    Language::parse(language_str).ok_or_else(|| {
        CmdError::validation(format!("Unsupported language '{language_str}' ({context})"))
    })
}

/// Gracefully terminate a process group: SIGTERM → wait → SIGKILL.
///
/// Implements Kubernetes-style graceful shutdown:
/// 1. Send SIGTERM to entire process group (allows cleanup)
/// 2. Wait for grace period
/// 3. If still running, send SIGKILL
pub(crate) async fn graceful_terminate_process_group(
    child: &mut tokio::process::Child,
    grace_period_secs: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::time::{Duration, timeout};

    let pid = match child.id() {
        Some(id) => id as i32,
        None => return Ok(()), // Process already exited
    };

    // Phase 1: SIGTERM to entire process group
    let term_result = unsafe { libc::kill(-pid, libc::SIGTERM) };
    if term_result == -1 {
        let errno = std::io::Error::last_os_error();
        if errno.raw_os_error() != Some(libc::ESRCH) {
            eprintln!("SIGTERM to process group {pid} failed: {errno}");
        }
        let _ = child.wait().await;
        return Ok(());
    }

    // Phase 2: Wait for grace period
    match timeout(Duration::from_secs(grace_period_secs), child.wait()).await {
        Ok(Ok(_)) => return Ok(()),
        Ok(Err(e)) => eprintln!("Wait error after SIGTERM: {e}"),
        Err(_) => eprintln!(
            "Process {pid} didn't respond to SIGTERM within {grace_period_secs}s, sending SIGKILL"
        ),
    }

    // Phase 3: SIGKILL
    let kill_result = unsafe { libc::kill(-pid, libc::SIGKILL) };
    if kill_result == -1 {
        let errno = std::io::Error::last_os_error();
        if errno.raw_os_error() != Some(libc::ESRCH) {
            eprintln!("SIGKILL to process group {pid} failed: {errno}");
        }
    }

    let _ = child.wait().await;
    Ok(())
}

/// Spawn a task that reads lines from a child process stream, respecting a byte limit.
///
/// Used by `install_packages` to capture stdout/stderr without duplication.
pub(crate) fn spawn_output_reader<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    stream: R,
    max_bytes: usize,
) -> tokio::task::JoinHandle<Vec<String>> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    tokio::spawn(async move {
        let mut reader = BufReader::new(stream).lines();
        let mut lines = Vec::new();
        let mut total_bytes = 0usize;

        while let Ok(Some(line)) = reader.next_line().await {
            if total_bytes + line.len() + 1 > max_bytes {
                let remaining = max_bytes.saturating_sub(total_bytes);
                if remaining > 0 {
                    lines.push(line[..remaining.min(line.len())].to_string());
                }
                lines.push(format!(
                    "[truncated: output limit {}KB exceeded]",
                    max_bytes / 1024
                ));
                break;
            }
            total_bytes += line.len() + 1;
            lines.push(line);
        }
        lines
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmd_error_validation() {
        let err = CmdError::validation("bad input");
        match err {
            CmdError::Reply {
                message,
                error_type,
                op_id,
            } => {
                assert_eq!(message, "bad input");
                assert_eq!(error_type, "validation_error");
                assert!(op_id.is_none());
            }
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_cmd_error_io_with_op_id() {
        let err = CmdError::io("disk full", Some("op1"));
        match err {
            CmdError::Reply {
                message,
                error_type,
                op_id,
            } => {
                assert_eq!(message, "disk full");
                assert_eq!(error_type, "io_error");
                assert_eq!(op_id.as_deref(), Some("op1"));
            }
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_cmd_error_timeout() {
        let err = CmdError::timeout("took too long");
        match err {
            CmdError::Reply { error_type, .. } => assert_eq!(error_type, "timeout_error"),
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_cmd_error_with_op_id_on_reply() {
        let err = CmdError::validation("msg").with_op_id("op1");
        match err {
            CmdError::Reply { op_id, .. } => assert_eq!(op_id.as_deref(), Some("op1")),
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_cmd_error_with_op_id_on_fatal_is_noop() {
        let err = CmdError::Fatal("boom".into());
        let err = err.with_op_id("op1");
        assert!(matches!(err, CmdError::Fatal(_)));
    }

    #[test]
    fn test_cmd_error_from_send_error() {
        let (tx, _rx) = mpsc::channel::<Vec<u8>>(1);
        drop(_rx);
        // Simulate a send error
        let _send_err = tx.try_send(vec![]).unwrap_err();
        // We can't directly convert TrySendError, but test the From impl exists
        let _: CmdError = mpsc::error::SendError(vec![]).into();
    }

    #[test]
    fn test_cmd_error_display_reply() {
        let err = CmdError::validation("bad input");
        assert_eq!(format!("{err}"), "bad input");
    }

    #[test]
    fn test_cmd_error_display_fatal() {
        let err = CmdError::Fatal("connection lost".into());
        assert_eq!(format!("{err}"), "fatal: connection lost");
    }

    #[test]
    fn test_parse_language_valid() {
        assert!(parse_language("python", "test").is_ok());
        assert!(parse_language("javascript", "test").is_ok());
        assert!(parse_language("raw", "test").is_ok());
    }

    #[test]
    fn test_parse_language_invalid() {
        let err = parse_language("cobol", "test").unwrap_err();
        match err {
            CmdError::Reply { message, .. } => {
                assert!(message.contains("cobol"));
            }
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }
}

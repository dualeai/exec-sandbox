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
#[derive(Debug)]
pub(crate) enum CmdError {
    Reply {
        message: String,
        error_type: &'static str,
    },
    Fatal(Box<dyn std::error::Error + Send + Sync>),
}

impl CmdError {
    pub(crate) fn validation(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: "validation_error",
        }
    }

    pub(crate) fn io(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: "io_error",
        }
    }

    pub(crate) fn timeout(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: "timeout_error",
        }
    }

    pub(crate) fn execution(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: "execution_error",
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
// ResponseWriter — op_id-aware response sender
// ============================================================================

/// Wraps a write channel and an optional `op_id`.  Every message sent through
/// this writer gets `op_id` injected into its JSON (unless the message already
/// carries one, e.g. file-transfer responses with their own `op_id`).
pub(crate) struct ResponseWriter {
    write_tx: mpsc::Sender<Vec<u8>>,
    op_id: Option<String>,
}

impl ResponseWriter {
    pub(crate) fn new(write_tx: mpsc::Sender<Vec<u8>>, op_id: Option<String>) -> Self {
        Self { write_tx, op_id }
    }

    /// Serialize `msg` to JSON, inject `op_id` (preserving any existing one),
    /// append newline, and queue for sending.
    pub(crate) async fn send(&self, msg: &GuestResponse) -> Result<(), CmdError> {
        let mut value = serde_json::to_value(msg).map_err(|e| CmdError::Fatal(e.into()))?;

        // Inject op_id only if the message doesn't already have one
        if let Some(ref id) = self.op_id
            && let Some(obj) = value.as_object_mut()
        {
            obj.entry("op_id")
                .or_insert_with(|| serde_json::Value::String(id.clone()));
        }

        let mut bytes = serde_json::to_vec(&value).map_err(|e| CmdError::Fatal(e.into()))?;
        bytes.push(b'\n');
        self.write_tx.send(bytes).await?;
        Ok(())
    }

    /// Build and send an Error response using the writer's `op_id`.
    pub(crate) async fn send_error(
        &self,
        message: String,
        error_type: &str,
    ) -> Result<(), CmdError> {
        self.send(&GuestResponse::Error {
            message,
            error_type: error_type.to_string(),
            op_id: self.op_id.clone(),
            version: Some(VERSION),
        })
        .await
    }
}

// ============================================================================
// Response helpers (pre-parse error paths where no op_id is available)
// ============================================================================

/// Serialize a GuestResponse to JSON + newline and queue it for sending.
///
/// Used only for pre-parse error paths (before `op_id` is extracted from JSON).
/// Prefer `ResponseWriter::send()` for all other cases.
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
///
/// Used only for pre-parse error paths (before `op_id` is extracted from JSON).
/// Prefer `ResponseWriter::send_error()` for all other cases.
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
            log_error!("SIGTERM to process group {pid} failed: {errno}");
        }
        let _ = child.wait().await;
        return Ok(());
    }

    // Phase 2: Wait for grace period
    match timeout(Duration::from_secs(grace_period_secs), child.wait()).await {
        Ok(Ok(_)) => return Ok(()),
        Ok(Err(e)) => log_error!("Wait error after SIGTERM: {e}"),
        Err(_) => log_warn!(
            "Process {pid} didn't respond to SIGTERM within {grace_period_secs}s, sending SIGKILL"
        ),
    }

    // Phase 3: SIGKILL
    let kill_result = unsafe { libc::kill(-pid, libc::SIGKILL) };
    if kill_result == -1 {
        let errno = std::io::Error::last_os_error();
        if errno.raw_os_error() != Some(libc::ESRCH) {
            log_error!("SIGKILL to process group {pid} failed: {errno}");
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
            } => {
                assert_eq!(message, "bad input");
                assert_eq!(error_type, "validation_error");
            }
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_cmd_error_io() {
        let err = CmdError::io("disk full");
        match err {
            CmdError::Reply {
                message,
                error_type,
            } => {
                assert_eq!(message, "disk full");
                assert_eq!(error_type, "io_error");
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

    // -------------------------------------------------------------------------
    // ResponseWriter
    // -------------------------------------------------------------------------

    /// Helper: send a message through ResponseWriter and return the JSON value.
    async fn writer_send_json(op_id: Option<&str>, msg: &GuestResponse) -> serde_json::Value {
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(8);
        let writer = ResponseWriter::new(tx, op_id.map(String::from));
        writer.send(msg).await.unwrap();
        let bytes = rx.recv().await.unwrap();
        // strip trailing newline
        let json_bytes = &bytes[..bytes.len() - 1];
        serde_json::from_slice(json_bytes).unwrap()
    }

    #[tokio::test]
    async fn response_writer_inject_op_id_into_pong() {
        let val = writer_send_json(Some("abc"), &GuestResponse::Pong { version: "1.0" }).await;
        assert_eq!(val["op_id"], "abc");
        assert_eq!(val["type"], "pong");
    }

    #[tokio::test]
    async fn response_writer_inject_op_id_into_stdout() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::Stdout {
                chunk: "hello".into(),
            },
        )
        .await;
        assert_eq!(val["op_id"], "abc");
    }

    #[tokio::test]
    async fn response_writer_inject_op_id_into_stderr() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::Stderr {
                chunk: "oops".into(),
            },
        )
        .await;
        assert_eq!(val["op_id"], "abc");
    }

    #[tokio::test]
    async fn response_writer_inject_op_id_into_complete() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::Complete {
                exit_code: 0,
                execution_time_ms: 100,
                spawn_ms: None,
                process_ms: None,
            },
        )
        .await;
        assert_eq!(val["op_id"], "abc");
    }

    #[tokio::test]
    async fn response_writer_inject_op_id_into_file_list() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::FileList {
                path: "".into(),
                entries: vec![],
            },
        )
        .await;
        assert_eq!(val["op_id"], "abc");
    }

    #[tokio::test]
    async fn response_writer_inject_op_id_into_warm_repl_ack() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::WarmReplAck {
                language: "python".into(),
                status: "ok".into(),
                message: None,
            },
        )
        .await;
        assert_eq!(val["op_id"], "abc");
    }

    #[tokio::test]
    async fn response_writer_preserve_existing_file_write_ack_op_id() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::FileWriteAck {
                op_id: "orig".into(),
                path: "f.txt".into(),
                bytes_written: 42,
            },
        )
        .await;
        // Existing op_id preserved, not overwritten
        assert_eq!(val["op_id"], "orig");
    }

    #[tokio::test]
    async fn response_writer_preserve_existing_file_chunk_op_id() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::FileChunk {
                op_id: "orig".into(),
                data: "SGVsbG8=".into(),
            },
        )
        .await;
        assert_eq!(val["op_id"], "orig");
    }

    #[tokio::test]
    async fn response_writer_preserve_existing_file_read_complete_op_id() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::FileReadComplete {
                op_id: "orig".into(),
                path: "f.txt".into(),
                size: 100,
            },
        )
        .await;
        assert_eq!(val["op_id"], "orig");
    }

    #[tokio::test]
    async fn response_writer_preserve_existing_error_op_id() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::Error {
                message: "fail".into(),
                error_type: "io_error".into(),
                op_id: Some("orig".into()),
                version: Some(VERSION),
            },
        )
        .await;
        assert_eq!(val["op_id"], "orig");
    }

    #[tokio::test]
    async fn response_writer_inject_into_error_without_op_id() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::Error {
                message: "fail".into(),
                error_type: "io_error".into(),
                op_id: None,
                version: Some(VERSION),
            },
        )
        .await;
        assert_eq!(val["op_id"], "abc");
    }

    #[tokio::test]
    async fn response_writer_no_op_id_writer_produces_no_op_id() {
        let val = writer_send_json(None, &GuestResponse::Pong { version: "1.0" }).await;
        assert!(val.get("op_id").is_none());
    }

    #[tokio::test]
    async fn response_writer_send_error() {
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(8);
        let writer = ResponseWriter::new(tx, Some("abc".into()));
        writer.send_error("boom".into(), "io_error").await.unwrap();
        let bytes = rx.recv().await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes[..bytes.len() - 1]).unwrap();
        assert_eq!(val["type"], "error");
        assert_eq!(val["message"], "boom");
        assert_eq!(val["error_type"], "io_error");
        assert_eq!(val["op_id"], "abc");
    }

    #[tokio::test]
    async fn response_writer_special_chars_in_op_id() {
        let val = writer_send_json(Some("a\"b\\c"), &GuestResponse::Pong { version: "1.0" }).await;
        assert_eq!(val["op_id"], "a\"b\\c");
    }

    #[tokio::test]
    async fn response_writer_unicode_op_id() {
        let val = writer_send_json(Some("日本語"), &GuestResponse::Pong { version: "1.0" }).await;
        assert_eq!(val["op_id"], "日本語");
    }

    #[tokio::test]
    async fn response_writer_long_op_id() {
        let long_id = "x".repeat(1000);
        let val = writer_send_json(Some(&long_id), &GuestResponse::Pong { version: "1.0" }).await;
        assert_eq!(val["op_id"].as_str().unwrap().len(), 1000);
    }

    #[tokio::test]
    async fn response_writer_empty_op_id() {
        let val = writer_send_json(Some(""), &GuestResponse::Pong { version: "1.0" }).await;
        assert_eq!(val["op_id"], "");
    }
}

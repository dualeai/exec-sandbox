//! Error types and response helpers for the guest agent.

use tokio::sync::mpsc;

use crate::constants::KILL_WAIT;
use crate::types::GuestResponse;

// ============================================================================
// ErrorType — domain-specific error classification
// ============================================================================

/// Domain-specific error types sent to the host as the `error_type` wire field.
///
/// Each variant maps to a distinct wire string, so the host can dispatch
/// without parsing messages. The host enum is `GuestErrorType` in
/// src/exec_sandbox/constants.py; keep the two in sync.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ErrorType {
    EnvVar,      // "env_var_error"
    Code,        // "code_error"
    Path,        // "path_error"
    Package,     // "package_error"
    Io,          // "io_error"
    Timeout,     // "timeout_error"
    Execution,   // "execution_error"
    Request,     // "request_error"
    Protocol,    // "protocol_error"
    OutputLimit, // "output_limit_error"
}

impl ErrorType {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::EnvVar => "env_var_error",
            Self::Code => "code_error",
            Self::Path => "path_error",
            Self::Package => "package_error",
            Self::Io => "io_error",
            Self::Timeout => "timeout_error",
            Self::Execution => "execution_error",
            Self::Request => "request_error",
            Self::Protocol => "protocol_error",
            Self::OutputLimit => "output_limit_error",
        }
    }
}

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
    pub(crate) fn env_var(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: ErrorType::EnvVar.as_str(),
        }
    }

    pub(crate) fn code(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: ErrorType::Code.as_str(),
        }
    }

    pub(crate) fn path(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: ErrorType::Path.as_str(),
        }
    }

    pub(crate) fn package(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: ErrorType::Package.as_str(),
        }
    }

    pub(crate) fn io(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: ErrorType::Io.as_str(),
        }
    }

    pub(crate) fn timeout(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: ErrorType::Timeout.as_str(),
        }
    }

    pub(crate) fn execution(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: ErrorType::Execution.as_str(),
        }
    }

    pub(crate) fn output_limit(msg: impl Into<String>) -> Self {
        Self::Reply {
            message: msg.into(),
            error_type: ErrorType::OutputLimit.as_str(),
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

/// Wraps a write channel and the optional `op_id` of a request. Every message
/// sent through this writer gets that `op_id` in its JSON.
pub(crate) struct ResponseWriter {
    write_tx: mpsc::Sender<Vec<u8>>,
    op_id: Option<String>,
}

impl ResponseWriter {
    pub(crate) fn new(write_tx: mpsc::Sender<Vec<u8>>, op_id: Option<String>) -> Self {
        Self { write_tx, op_id }
    }

    /// Serialize `msg` to JSON, inject `op_id`, append newline, and queue for sending.
    pub(crate) async fn send(&self, msg: &GuestResponse) -> Result<(), CmdError> {
        let mut value = serde_json::to_value(msg).map_err(|e| CmdError::Fatal(e.into()))?;

        if let Some(ref id) = self.op_id
            && let Some(obj) = value.as_object_mut()
        {
            obj.insert("op_id".to_string(), serde_json::Value::String(id.clone()));
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
        })
        .await
    }
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

/// Gracefully terminate a process group: SIGTERM → wait → SIGKILL.
///
/// Implements Kubernetes-style graceful shutdown:
/// 1. Send SIGTERM to entire process group (allows cleanup)
/// 2. Wait for the leader, at most the grace period
/// 3. Send SIGKILL to the group: as when the init of a container ends, the
///    members that outlive the leader (one that ignores SIGTERM) end with it
pub(crate) async fn graceful_terminate_process_group(
    child: &mut crate::init::WaitedChild,
    grace_period_secs: u64,
) {
    use tokio::time::{Duration, timeout};

    let pid = match child.id() {
        Some(id) => id as i32,
        None => return, // Process already exited (its PID may be another process now)
    };

    // Phase 1: SIGTERM to entire process group
    let term_result = unsafe { libc::kill(-pid, libc::SIGTERM) };
    if term_result == -1 {
        let errno = std::io::Error::last_os_error();
        if errno.raw_os_error() != Some(libc::ESRCH) {
            log_error!("SIGTERM to process group {pid} failed: {errno}");
        }
        let _ = wait_child_bounded(child, "a failed SIGTERM", KILL_WAIT).await;
        return;
    }

    // Phase 2: Wait for the leader, at most the grace period
    let leader_ended = match timeout(Duration::from_secs(grace_period_secs), child.wait()).await {
        Ok(Ok(_)) => true,
        Ok(Err(e)) => {
            log_error!("Wait error after SIGTERM: {e}");
            false
        }
        Err(_) => {
            log_warn!(
                "Process {pid} didn't respond to SIGTERM within {grace_period_secs}s, sending SIGKILL"
            );
            false
        }
    };

    // Phase 3: SIGKILL to the group, also after the leader ended. The group ID
    // stays reserved while a member lives, so the signal reaches only the
    // group (ESRCH when no member is left).
    let kill_result = unsafe { libc::kill(-pid, libc::SIGKILL) };
    if kill_result == -1 {
        let errno = std::io::Error::last_os_error();
        if errno.raw_os_error() != Some(libc::ESRCH) {
            log_error!("SIGKILL to process group {pid} failed: {errno}");
        }
    }

    if !leader_ended {
        let _ = wait_child_bounded(child, "SIGKILL", KILL_WAIT).await;
    }
}

/// Wait for `child` to exit, at most `limit` (`KILL_WAIT` in the agent).
///
/// For a child that got SIGKILL, or that closed both output pipes. A process
/// in uninterruptible sleep (D state) exits only when its I/O ends, and the
/// agent serves one command at a time: it must not wait for that. A child
/// that is still running is reaped later (tokio's orphan queue, or the PID 1
/// reaper).
pub(crate) async fn wait_child_bounded(
    child: &mut crate::init::WaitedChild,
    after: &str,
    limit: std::time::Duration,
) -> Option<std::process::ExitStatus> {
    let pid = child.id();
    match tokio::time::timeout(limit, child.wait()).await {
        Ok(Ok(status)) => Some(status),
        Ok(Err(e)) => {
            log_error!("Wait for process {pid:?} after {after} failed: {e}");
            None
        }
        Err(_) => {
            log_warn!("Process {pid:?} still runs {limit:?} after {after}; not waiting longer");
            None
        }
    }
}

/// Spawn a task that reads lines from a child process stream, respecting a byte limit.
///
/// Used by `install_packages` to collect stdout and stderr. Whole lines are
/// kept while their total fits in `max_bytes` (a newline counts one byte); the
/// line that crosses it is cut at a character boundary and a `[truncated: ...]`
/// line follows. The task then reads on and drops the rest, so the child never
/// writes into a closed pipe (EPIPE). Bytes that are not UTF-8 become U+FFFD.
/// A line holds at most `max_bytes` in memory, however long the child makes it.
pub(crate) fn spawn_output_reader<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    stream: R,
    max_bytes: usize,
) -> tokio::task::JoinHandle<Vec<String>> {
    use tokio::io::{AsyncBufReadExt, BufReader};

    tokio::spawn(async move {
        let mut reader = BufReader::new(stream);
        let mut lines = Vec::new();
        let mut kept = 0usize; // bytes of `lines`, a newline counting one
        let mut current: Vec<u8> = Vec::new();
        let mut truncated = false;
        loop {
            let chunk = match reader.fill_buf().await {
                Ok([]) | Err(_) => break, // end of the stream, or the child's end is gone
                Ok(chunk) => chunk,
            };
            let read = chunk.len();
            if !truncated {
                for &byte in chunk {
                    if byte == b'\n' {
                        truncated = keep_line(&mut lines, &mut kept, &mut current, max_bytes);
                        if truncated {
                            break;
                        }
                    } else if current.len() <= max_bytes {
                        current.push(byte); // past max_bytes the line is cut anyway
                    }
                }
            }
            reader.consume(read);
        }
        if !truncated && !current.is_empty() {
            keep_line(&mut lines, &mut kept, &mut current, max_bytes);
        }
        lines
    })
}

/// Add the line in `current` (its newline not included) to `lines`, or its
/// part that fits in `max_bytes` and a `[truncated: ...]` line. Returns true
/// once truncated: the caller then drops the rest of the stream.
fn keep_line(
    lines: &mut Vec<String>,
    kept: &mut usize,
    current: &mut Vec<u8>,
    max_bytes: usize,
) -> bool {
    if current.last() == Some(&b'\r') {
        current.pop();
    }
    let line = String::from_utf8_lossy(current).into_owned();
    current.clear();
    if *kept + line.len() < max_bytes {
        *kept += line.len() + 1;
        lines.push(line);
        return false;
    }
    let mut cut = max_bytes.saturating_sub(*kept).min(line.len());
    while !line.is_char_boundary(cut) {
        cut -= 1;
    }
    if cut > 0 {
        lines.push(line[..cut].to_string());
    }
    lines.push(format!(
        "[truncated: output limit {}KB exceeded]",
        max_bytes / 1024
    ));
    true
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {

    // -------------------------------------------------------------------------
    // spawn_output_reader
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_output_reader_keeps_whole_lines_then_marks_the_cut() {
        // "abcdeé" is 7 bytes and its newline 1: it fills the 8-byte limit, so
        // "xyz" goes and the truncation line follows
        let (mut tx, rx) = tokio::io::duplex(64);
        let task = spawn_output_reader(rx, 8);
        tokio::io::AsyncWriteExt::write_all(&mut tx, "abcdeé\nxyz\n".as_bytes())
            .await
            .unwrap();
        drop(tx);
        let lines = task.await.unwrap();
        assert_eq!(
            lines,
            vec![
                "abcdeé".to_string(),
                "[truncated: output limit 0KB exceeded]".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn test_output_reader_cut_inside_a_char_does_not_panic() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let task = spawn_output_reader(rx, 4);
        tokio::io::AsyncWriteExt::write_all(&mut tx, "abcé\n".as_bytes())
            .await
            .unwrap();
        drop(tx);
        let lines = task.await.unwrap();
        assert_eq!(
            lines,
            vec![
                "abc".to_string(),
                "[truncated: output limit 0KB exceeded]".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn test_output_reader_drains_past_the_limit() {
        // The child writes 64 KiB through a 1 KiB pipe after the limit: every
        // write must succeed (a closed pipe would give the child EPIPE).
        let (mut tx, rx) = tokio::io::duplex(1024);
        let task = spawn_output_reader(rx, 16);
        for _ in 0..1024 {
            tokio::io::AsyncWriteExt::write_all(&mut tx, &[b'x'; 63])
                .await
                .unwrap();
            tokio::io::AsyncWriteExt::write_all(&mut tx, b"\n")
                .await
                .unwrap();
        }
        drop(tx);
        let lines = task.await.unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "x".repeat(16));
    }

    #[tokio::test]
    async fn test_output_reader_keeps_reading_after_bytes_that_are_not_utf8() {
        let (mut tx, rx) = tokio::io::duplex(64);
        let task = spawn_output_reader(rx, 1024);
        tokio::io::AsyncWriteExt::write_all(&mut tx, b"a\xffb\r\nnext\nlast")
            .await
            .unwrap();
        drop(tx);
        let lines = task.await.unwrap();
        assert_eq!(
            lines,
            vec![
                "a\u{fffd}b".to_string(),
                "next".to_string(),
                "last".to_string()
            ]
        );
    }

    use super::*;

    #[test]
    fn exit_code_maps_signals_and_codes() {
        use std::os::unix::process::ExitStatusExt;
        use std::process::ExitStatus;

        // Raw wait status: low byte = signal number, code<<8 = normal exit.
        assert_eq!(exit_code_from_status(ExitStatus::from_raw(9)), 137); // SIGKILL
        assert_eq!(exit_code_from_status(ExitStatus::from_raw(11)), 139); // SIGSEGV
        assert_eq!(exit_code_from_status(ExitStatus::from_raw(0)), 0);
        assert_eq!(exit_code_from_status(ExitStatus::from_raw(1 << 8)), 1);
        assert_eq!(exit_code_from_status(ExitStatus::from_raw(137 << 8)), 137);
    }

    #[test]
    fn test_cmd_error_env_var() {
        let err = CmdError::env_var("blocked env var");
        match err {
            CmdError::Reply {
                message,
                error_type,
            } => {
                assert_eq!(message, "blocked env var");
                assert_eq!(error_type, "env_var_error");
            }
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_cmd_error_code() {
        let err = CmdError::code("code too large");
        match err {
            CmdError::Reply {
                message,
                error_type,
            } => {
                assert_eq!(message, "code too large");
                assert_eq!(error_type, "code_error");
            }
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_cmd_error_path() {
        let err = CmdError::path("path traversal");
        match err {
            CmdError::Reply {
                message,
                error_type,
            } => {
                assert_eq!(message, "path traversal");
                assert_eq!(error_type, "path_error");
            }
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_cmd_error_package() {
        let err = CmdError::package("invalid package name");
        match err {
            CmdError::Reply {
                message,
                error_type,
            } => {
                assert_eq!(message, "invalid package name");
                assert_eq!(error_type, "package_error");
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
    fn test_cmd_error_execution() {
        let err = CmdError::execution("spawn failed");
        match err {
            CmdError::Reply { error_type, .. } => assert_eq!(error_type, "execution_error"),
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_cmd_error_output_limit() {
        let err = CmdError::output_limit("stdout 1200000 bytes exceeds 1000000 limit");
        match err {
            CmdError::Reply {
                message,
                error_type,
            } => {
                assert!(message.contains("1200000"));
                assert_eq!(error_type, "output_limit_error");
            }
            CmdError::Fatal(_) => panic!("expected Reply"),
        }
    }

    #[test]
    fn test_error_type_as_str() {
        assert_eq!(ErrorType::EnvVar.as_str(), "env_var_error");
        assert_eq!(ErrorType::Code.as_str(), "code_error");
        assert_eq!(ErrorType::Path.as_str(), "path_error");
        assert_eq!(ErrorType::Package.as_str(), "package_error");
        assert_eq!(ErrorType::Io.as_str(), "io_error");
        assert_eq!(ErrorType::Timeout.as_str(), "timeout_error");
        assert_eq!(ErrorType::Execution.as_str(), "execution_error");
        assert_eq!(ErrorType::Request.as_str(), "request_error");
        assert_eq!(ErrorType::Protocol.as_str(), "protocol_error");
        assert_eq!(ErrorType::OutputLimit.as_str(), "output_limit_error");
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
        let err = CmdError::code("bad input");
        assert_eq!(format!("{err}"), "bad input");
    }

    #[test]
    fn test_cmd_error_display_fatal() {
        let err = CmdError::Fatal("connection lost".into());
        assert_eq!(format!("{err}"), "fatal: connection lost");
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

    const PONG: GuestResponse = GuestResponse::Pong { version: "1.0" };

    #[tokio::test]
    async fn response_writer_inject_op_id_into_pong() {
        let val = writer_send_json(Some("abc"), &PONG).await;
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
        let val = writer_send_json(Some("abc"), &GuestResponse::FileList { entries: vec![] }).await;
        assert_eq!(val["op_id"], "abc");
    }

    #[tokio::test]
    async fn response_writer_inject_op_id_into_warm_repl_ack() {
        let val = writer_send_json(
            Some("abc"),
            &GuestResponse::WarmReplAck {
                language: crate::types::Language::Python,
                status: "ok".into(),
                message: None,
            },
        )
        .await;
        assert_eq!(val["op_id"], "abc");
    }

    #[tokio::test]
    async fn response_writer_inject_op_id_into_file_transfer_messages() {
        // The host matches each chunk and the end of a transfer by this op_id
        for msg in [
            GuestResponse::FileWriteAck {
                path: "f.txt".into(),
                bytes_written: 42,
            },
            GuestResponse::FileChunk {
                data: "SGVsbG8=".into(),
            },
            GuestResponse::FileReadComplete {
                path: "f.txt".into(),
                size: 100,
            },
        ] {
            let val = writer_send_json(Some("abc"), &msg).await;
            assert_eq!(val["op_id"], "abc", "{msg:?}");
        }
    }

    #[tokio::test]
    async fn response_writer_no_op_id_writer_produces_no_op_id() {
        let val = writer_send_json(None, &PONG).await;
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
    async fn response_writer_error_of_a_request_with_no_op_id_has_none() {
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(8);
        let writer = ResponseWriter::new(tx, None);
        writer
            .send_error("bad".into(), "request_error")
            .await
            .unwrap();
        let bytes = rx.recv().await.unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes[..bytes.len() - 1]).unwrap();
        assert_eq!(
            val,
            serde_json::json!({"type": "error", "message": "bad", "error_type": "request_error"})
        );
    }

    #[tokio::test]
    async fn response_writer_special_chars_in_op_id() {
        let val = writer_send_json(Some("a\"b\\c"), &PONG).await;
        assert_eq!(val["op_id"], "a\"b\\c");
    }

    #[tokio::test]
    async fn response_writer_unicode_op_id() {
        let val = writer_send_json(Some("日本語"), &PONG).await;
        assert_eq!(val["op_id"], "日本語");
    }

    #[tokio::test]
    async fn response_writer_long_op_id() {
        let long_id = "x".repeat(1000);
        let val = writer_send_json(Some(&long_id), &PONG).await;
        assert_eq!(val["op_id"].as_str().unwrap().len(), 1000);
    }

    #[tokio::test]
    async fn response_writer_empty_op_id() {
        let val = writer_send_json(Some(""), &PONG).await;
        assert_eq!(val["op_id"], "");
    }

    // -------------------------------------------------------------------------
    // wait_child_bounded
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn wait_child_bounded_gives_up_on_a_child_that_keeps_running() {
        let mut child = crate::init::WaitedChild::register(
            tokio::process::Command::new("sleep")
                .arg("30")
                .spawn()
                .unwrap(),
        );
        let start = std::time::Instant::now();

        let status = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            wait_child_bounded(&mut child, "test", std::time::Duration::from_millis(50)),
        )
        .await
        .expect("the wait must stop at its limit");

        assert!(status.is_none());
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        child.start_kill().unwrap();
        child.wait().await.unwrap();
    }

    #[tokio::test]
    async fn graceful_terminate_ends_a_member_that_outlives_the_leader() {
        // The leader (sh) ends at SIGTERM; its child ignores SIGTERM and holds the
        // stdout pipe, so the pipe ends only when the group SIGKILL reaches it.
        let mut spawned = tokio::process::Command::new("sh")
            .args(["-c", "(trap '' TERM; echo ready; exec sleep 30) & wait"])
            .stdout(std::process::Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap();
        let mut stdout = spawned.stdout.take().unwrap();
        let mut child = crate::init::WaitedChild::register(spawned);
        let mut ready = [0u8; 6];
        tokio::io::AsyncReadExt::read_exact(&mut stdout, &mut ready)
            .await
            .unwrap();
        assert_eq!(&ready, b"ready\n"); // the member ignores SIGTERM from here

        graceful_terminate_process_group(&mut child, 5).await;

        let mut rest = Vec::new();
        let ended = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            tokio::io::AsyncReadExt::read_to_end(&mut stdout, &mut rest),
        )
        .await;
        assert!(
            matches!(ended, Ok(Ok(0))),
            "a member of the group still holds the pipe: {ended:?}"
        );
    }

    #[tokio::test]
    async fn wait_child_bounded_returns_the_status_of_a_child_that_exits() {
        let mut child = crate::init::WaitedChild::register(
            tokio::process::Command::new("sh")
                .args(["-c", "exit 3"])
                .spawn()
                .unwrap(),
        );

        let status =
            wait_child_bounded(&mut child, "test", std::time::Duration::from_secs(2)).await;

        assert_eq!(status.and_then(|s| s.code()), Some(3));
    }
}

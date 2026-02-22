//! Type definitions for the guest agent protocol.
//!
//! All structs for wire protocol messages (serde), internal state, and
//! pipeline handles live here to keep main.rs focused on logic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::mpsc;

// ============================================================================
// Inbound commands (host → guest, deserialized from JSON)
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(tag = "action")]
pub enum GuestCommand {
    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "install_packages")]
    InstallPackages {
        language: String,
        packages: Vec<String>,
    },

    #[serde(rename = "exec")]
    ExecuteCode {
        language: String,
        code: String,
        #[serde(default)]
        timeout: u64,
        #[serde(default)]
        env_vars: HashMap<String, String>,
    },

    #[serde(rename = "write_file")]
    WriteFile {
        op_id: String,
        path: String,
        #[serde(default)]
        make_executable: bool,
    },

    #[serde(rename = "read_file")]
    ReadFile { op_id: String, path: String },

    #[serde(rename = "file_chunk")]
    FileChunk { op_id: String, data: String },

    #[serde(rename = "file_end")]
    FileEnd { op_id: String },

    #[serde(rename = "list_files")]
    ListFiles {
        #[serde(default)]
        path: String,
    },

    #[serde(rename = "warm_repl")]
    WarmRepl { language: String },
}

// ============================================================================
// Outbound messages (guest → host, serialized to JSON)
// ============================================================================

/// All guest → host messages as a single tagged enum.
///
/// `#[serde(tag = "type")]` produces `{"type": "<variant>", ...}` — identical
/// wire format to the previous per-struct approach. The Python host discriminates
/// on `"type"` via Pydantic tagged union (`guest_agent_protocol.py`).
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum GuestResponse {
    #[serde(rename = "stdout")]
    Stdout { chunk: String },

    #[serde(rename = "stderr")]
    Stderr { chunk: String },

    #[serde(rename = "complete")]
    Complete {
        exit_code: i32,
        execution_time_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        spawn_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        process_ms: Option<u64>,
    },

    #[serde(rename = "pong")]
    Pong { version: &'static str },

    #[serde(rename = "error")]
    Error {
        message: String,
        error_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        op_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        version: Option<&'static str>,
    },

    #[serde(rename = "file_write_ack")]
    FileWriteAck {
        op_id: String,
        path: String,
        bytes_written: usize,
    },

    #[serde(rename = "file_chunk")]
    FileChunk { op_id: String, data: String },

    #[serde(rename = "file_read_complete")]
    FileReadComplete {
        op_id: String,
        path: String,
        size: u64,
    },

    #[serde(rename = "file_list")]
    FileList {
        path: String,
        entries: Vec<FileEntry>,
    },

    #[serde(rename = "warm_repl_ack")]
    WarmReplAck {
        language: String,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
}

#[derive(Debug, Serialize)]
pub struct FileEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

// ============================================================================
// File I/O internals
// ============================================================================

/// Wrapper that counts bytes written and enforces a size limit.
pub struct CountingWriter<W> {
    pub inner: W,
    pub count: usize,
    pub limit: usize,
}

impl<W: std::io::Write> std::io::Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.count + buf.len() > self.limit {
            return Err(std::io::Error::other("file size limit exceeded"));
        }
        let n = self.inner.write(buf)?;
        self.count += n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// State for an in-progress streaming file write operation.
/// Moved into a `spawn_blocking` task for pipelined decompression + disk I/O.
pub struct WriteFileState {
    pub decoder: zstd::stream::write::Decoder<'static, CountingWriter<std::fs::File>>,
    pub tmp_path: std::path::PathBuf,
    pub final_path: std::path::PathBuf,
    pub make_executable: bool,
    pub op_id: String,
    pub path_display: String, // relative path for error messages
    pub finished: bool,
}

impl WriteFileState {
    /// Build a `WriteError` for this write operation with `error_type = "io_error"`.
    pub fn io_error(&self, message: impl std::fmt::Display) -> WriteError {
        WriteError {
            message: format!("I/O error for '{}': {}", self.path_display, message),
            error_type: "io_error".to_string(),
            path_display: self.path_display.clone(),
            op_id: self.op_id.clone(),
        }
    }
}

impl Drop for WriteFileState {
    fn drop(&mut self) {
        // Clean up temp file if write was not completed (error/disconnect)
        if !self.finished {
            let _ = std::fs::remove_file(&self.tmp_path);
        }
    }
}

/// Handle for an in-progress pipelined file write.
/// Decouples chunk reception (main loop) from decompression+disk I/O (blocking thread pool).
pub struct ActiveWriteHandle {
    /// Send decoded (but not yet decompressed) chunks to the blocking worker.
    pub chunk_tx: mpsc::Sender<Vec<u8>>,
    /// Join handle for the blocking worker task.
    pub task: tokio::task::JoinHandle<Result<WriteResult, WriteError>>,
    /// For error messages when chunk_tx.send() fails.
    pub path_display: String,
    /// Correlates interleaved write operations on the orchestrator side.
    pub op_id: String,
}

pub struct WriteResult {
    pub bytes_written: usize,
    pub path_display: String,
    pub op_id: String,
}

pub struct WriteError {
    pub message: String,
    pub error_type: String,
    pub path_display: String,
    pub op_id: String,
}

// ============================================================================
// Language dispatch
// ============================================================================

/// Supported execution languages.
///
/// `GuestCommand` keeps `language: String` so unknown languages (e.g. `"cobol"`)
/// still deserialize — validation happens at the handler level via `Language::parse`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Python,
    Javascript,
    Raw,
}

impl Language {
    /// Parse a language string. Returns `None` for unsupported languages.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "python" => Some(Self::Python),
            "javascript" => Some(Self::Javascript),
            "raw" => Some(Self::Raw),
            _ => None,
        }
    }

    /// Wire-format string (matches `GuestCommand.language` values).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Javascript => "javascript",
            Self::Raw => "raw",
        }
    }
}

// ============================================================================
// REPL state
// ============================================================================

/// State for a persistent REPL process.
pub struct ReplState {
    pub child: tokio::process::Child,
    pub stdin: tokio::process::ChildStdin,
    pub stdout: tokio::process::ChildStdout,
    pub stderr: tokio::process::ChildStderr,
}

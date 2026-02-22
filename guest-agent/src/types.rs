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

#[derive(Debug, Serialize)]
pub struct OutputChunk {
    #[serde(rename = "type")]
    pub chunk_type: String, // "stdout" or "stderr"
    pub chunk: String,
}

#[derive(Debug, Serialize)]
pub struct ExecutionComplete {
    #[serde(rename = "type")]
    pub msg_type: String, // "complete"
    pub exit_code: i32,
    pub execution_time_ms: u64,
    /// Time for cmd.spawn() to return (fork/exec overhead)
    pub spawn_ms: Option<u64>,
    /// Time from spawn completion to child.wait() returning (actual process runtime)
    pub process_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct Pong {
    #[serde(rename = "type")]
    pub msg_type: String, // "pong"
    pub version: String,
}

#[derive(Debug, Serialize)]
pub struct StreamingError {
    #[serde(rename = "type")]
    pub msg_type: String, // "error"
    pub message: String,
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct FileWriteAck {
    #[serde(rename = "type")]
    pub msg_type: String, // "file_write_ack"
    pub op_id: String,
    pub path: String,
    pub bytes_written: usize,
}

#[derive(Debug, Serialize)]
pub struct FileChunkResponse {
    #[serde(rename = "type")]
    pub msg_type: String, // "file_chunk"
    pub op_id: String,
    pub data: String,
}

#[derive(Debug, Serialize)]
pub struct FileReadComplete {
    #[serde(rename = "type")]
    pub msg_type: String, // "file_read_complete"
    pub op_id: String,
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Serialize)]
pub struct FileEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

#[derive(Debug, Serialize)]
pub struct FileList {
    #[serde(rename = "type")]
    pub msg_type: String, // "file_list"
    pub path: String,
    pub entries: Vec<FileEntry>,
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
// REPL state
// ============================================================================

/// State for a persistent REPL process.
pub struct ReplState {
    pub child: tokio::process::Child,
    pub stdin: tokio::process::ChildStdin,
    pub stdout: tokio::process::ChildStdout,
    pub stderr: tokio::process::ChildStderr,
}

//! File I/O: read, write, list operations with sandbox path validation.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tokio::sync::mpsc;

use crate::constants::*;
use crate::error::{CmdError, ResponseWriter};
use crate::types::{FileEntry, GuestResponse};

// ============================================================================
// File I/O internals (moved from types.rs)
// ============================================================================

/// Wrapper that counts bytes written and enforces a size limit.
pub(crate) struct CountingWriter<W> {
    pub(crate) inner: W,
    pub(crate) count: usize,
    pub(crate) limit: usize,
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
pub(crate) struct WriteFileState {
    pub(crate) decoder: zstd::stream::write::Decoder<'static, CountingWriter<std::fs::File>>,
    pub(crate) tmp_path: std::path::PathBuf,
    pub(crate) final_path: std::path::PathBuf,
    pub(crate) make_executable: bool,
    pub(crate) op_id: String,
    pub(crate) path_display: String,
    pub(crate) finished: bool,
}

impl WriteFileState {
    pub(crate) fn io_error(&self, message: impl std::fmt::Display) -> WriteError {
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
        if !self.finished {
            let _ = std::fs::remove_file(&self.tmp_path);
        }
    }
}

/// Handle for an in-progress pipelined file write.
pub(crate) struct ActiveWriteHandle {
    pub(crate) chunk_tx: mpsc::Sender<Vec<u8>>,
    pub(crate) task: tokio::task::JoinHandle<Result<WriteResult, WriteError>>,
    pub(crate) path_display: String,
}

pub(crate) struct WriteResult {
    pub(crate) bytes_written: usize,
    pub(crate) path_display: String,
    pub(crate) op_id: String,
}

pub(crate) struct WriteError {
    pub(crate) message: String,
    pub(crate) error_type: String,
    pub(crate) path_display: String,
    pub(crate) op_id: String,
}

// ============================================================================
// Path validation
// ============================================================================

/// Validate a relative file path and resolve it under SANDBOX_ROOT.
pub(crate) fn validate_file_path(relative_path: &str) -> Result<std::path::PathBuf, String> {
    if relative_path.is_empty() {
        return Ok(std::path::PathBuf::from(SANDBOX_ROOT));
    }
    if relative_path.len() > MAX_FILE_PATH_LENGTH {
        return Err(format!(
            "Path too long: {} chars (max {})",
            relative_path.len(),
            MAX_FILE_PATH_LENGTH
        ));
    }
    if relative_path.contains('\0') {
        return Err("Path contains null byte".to_string());
    }
    if relative_path.chars().any(|c| c.is_control()) {
        return Err("Path contains control character".to_string());
    }
    if relative_path.starts_with('/') {
        return Err("Absolute paths are not allowed".to_string());
    }

    let sandbox_root = std::path::PathBuf::from(SANDBOX_ROOT);
    let mut resolved = sandbox_root.clone();

    for component in std::path::Path::new(relative_path).components() {
        match component {
            std::path::Component::Normal(c) => {
                let name_bytes = c.as_encoded_bytes().len();
                if name_bytes > MAX_FILE_NAME_BYTES {
                    return Err(format!(
                        "Filename component too long: {} bytes (max {})",
                        name_bytes, MAX_FILE_NAME_BYTES
                    ));
                }
                resolved.push(c);
            }
            std::path::Component::ParentDir => {
                if !resolved.pop() || !resolved.starts_with(&sandbox_root) {
                    return Err("Path traversal outside sandbox".to_string());
                }
            }
            std::path::Component::CurDir => {}
            _ => return Err("Invalid path component".to_string()),
        }
    }

    if !resolved.starts_with(&sandbox_root) {
        return Err("Path traversal outside sandbox".to_string());
    }

    Ok(resolved)
}

// ============================================================================
// File handlers
// ============================================================================

/// Handle read_file command: read from disk, stream as zstd-compressed chunks.
pub(crate) async fn handle_read_file(
    op_id: &str,
    path: &str,
    writer: &ResponseWriter,
) -> Result<(), CmdError> {
    let resolved_path = validate_file_path(path)
        .map_err(|e| CmdError::validation(format!("Invalid path '{path}': {e}")))?;

    if resolved_path == std::path::Path::new(SANDBOX_ROOT) {
        return Err(CmdError::validation("Cannot read a directory"));
    }

    let canonical_path = tokio::fs::canonicalize(&resolved_path)
        .await
        .map_err(|e| CmdError::io(format!("File not found or inaccessible '{path}': {e}")))?;

    let sandbox_canonical = tokio::fs::canonicalize(SANDBOX_ROOT)
        .await
        .unwrap_or_else(|_| std::path::PathBuf::from(SANDBOX_ROOT));
    if !canonical_path.starts_with(&sandbox_canonical) {
        return Err(CmdError::validation(format!(
            "Path '{path}' resolves outside sandbox"
        )));
    }

    let metadata = tokio::fs::metadata(&canonical_path)
        .await
        .map_err(|e| CmdError::io(format!("Cannot read '{path}': {e}")))?;

    if metadata.is_dir() {
        return Err(CmdError::validation(format!(
            "'{path}' is a directory, not a file"
        )));
    }
    if metadata.len() as usize > MAX_FILE_SIZE_BYTES {
        return Err(CmdError::validation(format!(
            "File too large: {} bytes (max {})",
            metadata.len(),
            MAX_FILE_SIZE_BYTES
        )));
    }

    let file_size = metadata.len();
    let file = std::fs::File::open(&canonical_path)
        .map_err(|e| CmdError::io(format!("Failed to read '{path}': {e}")))?;

    let mut encoder = zstd::stream::read::Encoder::new(file, FILE_TRANSFER_ZSTD_LEVEL)
        .map_err(|e| CmdError::io(format!("Compression init failed for '{path}': {e}")))?;

    let mut buf = vec![0u8; FILE_TRANSFER_CHUNK_SIZE];
    loop {
        let n = match std::io::Read::read(&mut encoder, &mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                return Err(CmdError::io(format!("Compression error: {e}")));
            }
        };
        let chunk_b64 = BASE64.encode(&buf[..n]);
        writer
            .send(&GuestResponse::FileChunk {
                op_id: op_id.to_string(),
                data: chunk_b64,
            })
            .await?;
    }

    writer
        .send(&GuestResponse::FileReadComplete {
            op_id: op_id.to_string(),
            path: path.to_string(),
            size: file_size,
        })
        .await
}

/// Handle list_files command: read directory entries.
pub(crate) async fn handle_list_files(path: &str, writer: &ResponseWriter) -> Result<(), CmdError> {
    let resolved_path = validate_file_path(path)
        .map_err(|e| CmdError::validation(format!("Invalid path '{path}': {e}")))?;

    let mut read_dir = tokio::fs::read_dir(&resolved_path)
        .await
        .map_err(|e| CmdError::io(format!("Cannot list '{path}': {e}")))?;

    let mut entries = Vec::new();
    while let Ok(Some(entry)) = read_dir.next_entry().await {
        let metadata = match entry.metadata().await {
            Ok(m) => m,
            Err(_) => continue,
        };
        entries.push(FileEntry {
            name: entry.file_name().to_string_lossy().to_string(),
            is_dir: metadata.is_dir(),
            size: if metadata.is_dir() { 0 } else { metadata.len() },
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));

    writer
        .send(&GuestResponse::FileList {
            path: path.to_string(),
            entries,
        })
        .await
}

/// Set up a pipelined file write operation.
///
/// Returns `Some(ActiveWriteHandle)` on success, or an error.
pub(crate) async fn handle_write_file(
    op_id: &str,
    path: &str,
    make_executable: bool,
) -> Result<ActiveWriteHandle, CmdError> {
    let resolved_path = validate_file_path(path)
        .map_err(|e| CmdError::validation(format!("Invalid path '{path}': {e}")))?;

    if resolved_path == std::path::Path::new(SANDBOX_ROOT) {
        return Err(CmdError::validation(
            "Cannot write to sandbox root directory",
        ));
    }

    if let Some(parent) = resolved_path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return Err(CmdError::io(format!("Failed to create directories: {e}")));
    }

    let tmp_path = resolved_path
        .parent()
        .unwrap_or(std::path::Path::new(SANDBOX_ROOT))
        .join(format!(".wr.{op_id}.tmp"));

    let file = std::fs::File::create(&tmp_path)
        .map_err(|e| CmdError::io(format!("Failed to create temp file: {e}")))?;

    let counting_writer = CountingWriter {
        inner: file,
        count: 0,
        limit: MAX_FILE_SIZE_BYTES,
    };
    let decoder = match zstd::stream::write::Decoder::new(counting_writer) {
        Ok(d) => d,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(CmdError::io(format!("Decompression init failed: {e}")));
        }
    };

    let (chunk_tx, chunk_rx) = mpsc::channel::<Vec<u8>>(16);
    let write_state = WriteFileState {
        decoder,
        tmp_path,
        final_path: resolved_path,
        make_executable,
        op_id: op_id.to_string(),
        path_display: path.to_string(),
        finished: false,
    };

    let task = tokio::task::spawn_blocking(move || {
        let mut state = write_state;
        let mut chunk_rx = chunk_rx;
        let mut finalize = false;
        while let Some(decoded) = chunk_rx.blocking_recv() {
            if decoded.is_empty() {
                finalize = true;
                break;
            }
            if let Err(e) = std::io::Write::write_all(&mut state.decoder, &decoded) {
                return Err(state.io_error(format!("write: {e}")));
            }
        }
        if !finalize {
            return Err(state.io_error("channel closed without finalize"));
        }
        if let Err(e) = std::io::Write::flush(&mut state.decoder) {
            return Err(state.io_error(format!("decompression finalize: {e}")));
        }
        let bytes_written = state.decoder.get_ref().count;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = if state.make_executable { 0o755 } else { 0o644 };
            if let Err(e) =
                std::fs::set_permissions(&state.tmp_path, std::fs::Permissions::from_mode(mode))
            {
                return Err(state.io_error(format!("set_permissions: {e}")));
            }
            let ret = unsafe {
                libc::chown(
                    std::ffi::CString::new(state.tmp_path.as_os_str().as_encoded_bytes())
                        .unwrap()
                        .as_ptr(),
                    SANDBOX_UID,
                    SANDBOX_GID,
                )
            };
            if ret != 0 {
                let e = std::io::Error::last_os_error();
                return Err(state.io_error(format!("chown: {e}")));
            }
        }
        if let Err(e) = std::fs::rename(&state.tmp_path, &state.final_path) {
            return Err(state.io_error(format!("rename: {e}")));
        }
        state.finished = true;
        Ok(WriteResult {
            bytes_written,
            path_display: state.path_display.clone(),
            op_id: state.op_id.clone(),
        })
    });

    Ok(ActiveWriteHandle {
        chunk_tx,
        task,
        path_display: path.to_string(),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // validate_file_path
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_file_path_basic() {
        let result = validate_file_path("hello.txt");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/home/user/hello.txt")
        );
    }

    #[test]
    fn test_validate_file_path_nested() {
        let result = validate_file_path("subdir/data.csv");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/home/user/subdir/data.csv")
        );
    }

    #[test]
    fn test_validate_file_path_deeply_nested() {
        assert!(validate_file_path("a/b/c/d/e.txt").is_ok());
    }

    #[test]
    fn test_validate_file_path_traversal_rejected() {
        let result = validate_file_path("../etc/passwd");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("traversal"));
    }

    #[test]
    fn test_validate_file_path_mid_path_traversal() {
        assert!(validate_file_path("subdir/../../etc/shadow").is_err());
    }

    #[test]
    fn test_validate_file_path_deep_traversal() {
        assert!(validate_file_path("a/b/../../../etc/passwd").is_err());
    }

    #[test]
    fn test_validate_file_path_absolute_rejected() {
        let result = validate_file_path("/etc/passwd");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Absolute"));
    }

    #[test]
    fn test_validate_file_path_empty_returns_sandbox_root() {
        let result = validate_file_path("");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), std::path::PathBuf::from("/home/user"));
    }

    #[test]
    fn test_validate_file_path_null_byte_rejected() {
        let result = validate_file_path("file\x00.txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("null byte"));
    }

    #[test]
    fn test_validate_file_path_control_chars_rejected() {
        let result = validate_file_path("file\x01name");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("control character"));
    }

    #[test]
    fn test_validate_file_path_component_too_long() {
        let long_name = "a".repeat(256);
        let result = validate_file_path(&long_name);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too long"));
    }

    #[test]
    fn test_validate_file_path_component_exactly_max() {
        let name = "a".repeat(255);
        assert!(validate_file_path(&name).is_ok());
    }

    #[test]
    fn test_validate_file_path_dot_dot_normalization() {
        let result = validate_file_path("subdir/../file.txt");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/home/user/file.txt")
        );
    }

    #[test]
    fn test_validate_file_path_double_slash() {
        assert!(validate_file_path("dir//file.txt").is_ok());
    }

    #[test]
    fn test_validate_file_path_trailing_slash() {
        assert!(validate_file_path("subdir/").is_ok());
    }

    #[test]
    fn test_validate_file_path_hidden_file() {
        let result = validate_file_path(".hidden");
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            std::path::PathBuf::from("/home/user/.hidden")
        );
    }

    #[test]
    fn test_validate_file_path_unicode() {
        assert!(validate_file_path("日本語.txt").is_ok());
    }

    #[test]
    fn test_validate_file_path_dot_only() {
        let result = validate_file_path(".");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), std::path::PathBuf::from("/home/user"));
    }

    #[test]
    fn test_validate_file_path_exactly_max_length() {
        // Construct a valid long path with short-enough components
        let path: String = (0..16)
            .map(|_| "a".repeat(250))
            .collect::<Vec<_>>()
            .join("/");
        if path.len() <= MAX_FILE_PATH_LENGTH {
            assert!(validate_file_path(&path).is_ok());
        }
    }

    #[test]
    fn test_validate_file_path_exceeds_max_length() {
        let path = "a/".repeat(MAX_FILE_PATH_LENGTH);
        assert!(validate_file_path(&path).is_err());
    }

    // -------------------------------------------------------------------------
    // CountingWriter
    // -------------------------------------------------------------------------

    #[test]
    fn test_counting_writer_within_limit() {
        let mut buf = Vec::new();
        let mut writer = CountingWriter {
            inner: &mut buf,
            count: 0,
            limit: 100,
        };
        assert!(std::io::Write::write_all(&mut writer, b"hello").is_ok());
        assert_eq!(writer.count, 5);
    }

    #[test]
    fn test_counting_writer_at_limit() {
        let mut buf = Vec::new();
        let mut writer = CountingWriter {
            inner: &mut buf,
            count: 0,
            limit: 5,
        };
        assert!(std::io::Write::write_all(&mut writer, b"hello").is_ok());
        assert_eq!(writer.count, 5);
    }

    #[test]
    fn test_counting_writer_exceeds_limit() {
        let mut buf = Vec::new();
        let mut writer = CountingWriter {
            inner: &mut buf,
            count: 0,
            limit: 4,
        };
        let result = std::io::Write::write(&mut writer, b"hello");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("file size limit exceeded")
        );
    }

    #[test]
    fn test_counting_writer_multiple_writes_exceeding() {
        let mut buf = Vec::new();
        let mut writer = CountingWriter {
            inner: &mut buf,
            count: 0,
            limit: 8,
        };
        assert!(std::io::Write::write_all(&mut writer, b"hello").is_ok());
        let result = std::io::Write::write(&mut writer, b"world");
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Serde / zstd roundtrip
    // -------------------------------------------------------------------------

    #[test]
    fn test_zstd_roundtrip() {
        let data = vec![42u8; FILE_TRANSFER_CHUNK_SIZE];
        let compressed = zstd::stream::encode_all(&data[..], FILE_TRANSFER_ZSTD_LEVEL).unwrap();
        let decompressed = zstd::stream::decode_all(&compressed[..]).unwrap();
        assert_eq!(data, decompressed);
    }
}

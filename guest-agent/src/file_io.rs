//! File I/O: read, write, list operations with sandbox path validation.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tokio::sync::mpsc;

use crate::constants::*;
use crate::error::{CmdError, ResponseWriter};
use crate::types::{FileEntry, GuestResponse};

// ============================================================================
// File I/O internals
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
    /// The target's folder, opened beneath SANDBOX_ROOT (`make_dirs_beneath`): the
    /// temporary file is made, renamed and removed through it, never through a path.
    pub(crate) dir: OwnedFd,
    pub(crate) tmp_name: CString,
    pub(crate) final_name: CString,
    pub(crate) make_executable: bool,
    pub(crate) path_display: String,
    pub(crate) finished: bool,
}

impl WriteFileState {
    /// The message of the `io_error` reply for a failed write.
    pub(crate) fn io_error(&self, message: impl std::fmt::Display) -> String {
        format!("I/O error for '{}': {}", self.path_display, message)
    }
}

impl Drop for WriteFileState {
    fn drop(&mut self) {
        if !self.finished {
            // SAFETY: a valid folder descriptor and a NUL-terminated name.
            unsafe { libc::unlinkat(self.dir.as_raw_fd(), self.tmp_name.as_ptr(), 0) };
        }
    }
}

/// Handle for an in-progress pipelined file write.
pub(crate) struct ActiveWriteHandle {
    pub(crate) chunk_tx: mpsc::Sender<Vec<u8>>,
    /// Bytes written, or the message of an `io_error` reply (`run_write`).
    pub(crate) task: tokio::task::JoinHandle<Result<usize, String>>,
    pub(crate) path_display: String,
    /// Who ends the write: the task's commit or the agent's abandon at a stall.
    pub(crate) end: Arc<WriteEnd>,
}

/// How a pipelined write ends: its task commits it (the rename), or the agent
/// abandons it (a stall). One atomic claim decides, so an abandoned write never
/// lands, and a write that lands is never reported abandoned.
pub(crate) struct WriteEnd(AtomicU8);

impl WriteEnd {
    const OPEN: u8 = 0;
    const ABANDONED: u8 = 1;
    const COMMITTING: u8 = 2;

    pub(crate) fn new() -> Self {
        Self(AtomicU8::new(Self::OPEN))
    }

    /// The task, before its rename: true when it may commit; false when the
    /// agent abandoned the write first.
    pub(crate) fn try_commit(&self) -> bool {
        self.0
            .compare_exchange(
                Self::OPEN,
                Self::COMMITTING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// The agent, at a stall: true when the write is abandoned (its task will
    /// not commit); false when the task claimed the commit first (its rename is
    /// on its way).
    pub(crate) fn try_abandon(&self) -> bool {
        self.0
            .compare_exchange(
                Self::OPEN,
                Self::ABANDONED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

// ============================================================================
// Path validation
// ============================================================================

/// Validate a relative file path and resolve it under SANDBOX_ROOT.
///
/// An empty path is SANDBOX_ROOT. The check is lexical: it does not resolve
/// symlinks (`open_beneath` resolves them and checks where they lead).
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

    // `resolved` stays under SANDBOX_ROOT at each step (a ParentDir that would
    // leave it is refused): no check at the end.
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
                resolved.pop();
                if !resolved.starts_with(&sandbox_root) {
                    return Err("Path traversal outside sandbox".to_string());
                }
            }
            std::path::Component::CurDir => {}
            // RootDir and Prefix: no relative Unix path has them
            _ => return Err("Invalid path component".to_string()),
        }
    }

    Ok(resolved)
}

// ============================================================================
// Resolution beneath the sandbox root
// ============================================================================

/// The part of a `validate_file_path` result below SANDBOX_ROOT ("" for the root).
fn below_root(resolved: &Path) -> &Path {
    resolved
        .strip_prefix(SANDBOX_ROOT)
        .unwrap_or_else(|_| Path::new(""))
}

/// Open `rel` below the folder `root`, with `flags`. Links resolve as the
/// kernel resolves them, absolute ones too, so a link to another place in
/// `root` works (`ln -s "$PWD/run-3" latest`); the file reached must be `root`
/// or below it, else EXDEV. The agent runs as root: guest code could otherwise
/// plant a link in SANDBOX_ROOT that steers a host write, read or listing out
/// of it.
///
/// Steps: an O_PATH open of `root/rel` (openat2 RESOLVE_NO_MAGICLINKS: a
/// /proc/<pid>/fd/* link fails, ELOOP; O_PATH runs no driver open, so a link
/// to a device has no effect), the check of where that descriptor is
/// (/proc/self/fd/N), then, unless `flags` has O_PATH, a reopen of that same
/// file through /proc/self/fd/N with `flags`: no window between the check and
/// the open. `flags` must not hold O_CREAT or O_NOFOLLOW. `root` must be a
/// canonical path (SANDBOX_ROOT is): /proc/self/fd/N names the canonical one,
/// so any other root refuses every open (EXDEV). `rel` holds no NUL byte
/// (validate_file_path refuses it).
fn open_beneath(root: &Path, rel: &Path, flags: libc::c_int) -> std::io::Result<OwnedFd> {
    let full = root.join(rel);
    let c_full = CString::new(full.as_os_str().as_bytes()).expect("a path with no NUL byte");
    // SAFETY: open_how is plain data; zero is its default for every field.
    let mut how: libc::open_how = unsafe { std::mem::zeroed() };
    how.flags = (libc::O_PATH | libc::O_CLOEXEC | (flags & libc::O_DIRECTORY)) as u64;
    how.resolve = libc::RESOLVE_NO_MAGICLINKS;
    // SAFETY: NUL-terminated path, and the size of `how`.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            libc::AT_FDCWD,
            c_full.as_ptr(),
            &how as *const libc::open_how,
            std::mem::size_of::<libc::open_how>(),
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fd` is a new descriptor that nothing else owns.
    let path_fd = unsafe { OwnedFd::from_raw_fd(fd as libc::c_int) };
    let fd_link = format!("/proc/self/fd/{}", path_fd.as_raw_fd());
    if !std::fs::read_link(&fd_link)?.starts_with(root) {
        return Err(std::io::Error::from_raw_os_error(libc::EXDEV));
    }
    if flags & libc::O_PATH != 0 {
        return Ok(path_fd);
    }
    let c_link = CString::new(fd_link).expect("a /proc/self/fd path has no NUL byte");
    // SAFETY: a NUL-terminated path; the descriptor is owned at once.
    let fd = unsafe { libc::open(c_link.as_ptr(), flags | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fd` is a new descriptor that nothing else owns.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// The folder `rel_dir` below `root` as a path handle, its missing folders
/// made (mode 0755, owned by the sandbox user as the written file is, so guest
/// code can add files to them) one by one: each is made in its parent's
/// handle and opened with `open_beneath` again, so a link that guest code
/// plants on the way cannot lead the walk out of `root` (EXDEV). A link at a
/// missing folder's name is never followed by the make (mkdirat: EEXIST).
fn make_dirs_beneath(root: &Path, rel_dir: &Path) -> std::io::Result<OwnedFd> {
    let mut walked = std::path::PathBuf::new();
    let mut dir = open_beneath(root, &walked, libc::O_PATH | libc::O_DIRECTORY)?;
    for component in rel_dir.components() {
        walked.push(component);
        match open_beneath(root, &walked, libc::O_PATH | libc::O_DIRECTORY) {
            Ok(next) => dir = next,
            Err(e) if e.raw_os_error() == Some(libc::ENOENT) => {
                let name = CString::new(component.as_os_str().as_bytes())
                    .expect("a path with no NUL byte");
                // SAFETY: valid folder descriptor and NUL-terminated name.
                let made = unsafe { libc::mkdirat(dir.as_raw_fd(), name.as_ptr(), 0o755) } == 0;
                if !made {
                    let e = std::io::Error::last_os_error();
                    if e.raw_os_error() != Some(libc::EEXIST) {
                        return Err(e); // EEXIST: another write made it meanwhile
                    }
                }
                dir = open_beneath(root, &walked, libc::O_PATH | libc::O_DIRECTORY)?;
                if made {
                    // SAFETY: valid descriptor; an empty path with AT_EMPTY_PATH
                    // names that folder itself (no path to swap).
                    let owned = unsafe {
                        libc::fchownat(
                            dir.as_raw_fd(),
                            c"".as_ptr(),
                            SANDBOX_UID,
                            SANDBOX_GID,
                            libc::AT_EMPTY_PATH,
                        )
                    };
                    if owned != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
            }
            Err(e) => return Err(e),
        }
    }
    Ok(dir)
}

/// The command error of a failed `open_beneath`: a path error when the path
/// leaves SANDBOX_ROOT (EXDEV) or meets a link loop or a /proc link (ELOOP),
/// else an I/O error `what` (not found, no access, ...).
fn beneath_error(path: &str, e: &std::io::Error, what: &str) -> CmdError {
    match e.raw_os_error() {
        Some(libc::EXDEV) => CmdError::path(format!("Path '{path}' resolves outside sandbox")),
        Some(libc::ELOOP) => {
            CmdError::path(format!("Path '{path}' has a link loop or a /proc link"))
        }
        _ => CmdError::io(format!("{what} '{path}': {e}")),
    }
}

// ============================================================================
// File handlers
// ============================================================================

/// Handle read_file command: read from disk, stream as zstd-compressed chunks.
pub(crate) async fn handle_read_file(path: &str, writer: &ResponseWriter) -> Result<(), CmdError> {
    let resolved_path = validate_file_path(path)
        .map_err(|e| CmdError::path(format!("Invalid path '{path}': {e}")))?;

    if resolved_path == std::path::Path::new(SANDBOX_ROOT) {
        return Err(CmdError::path("Cannot read a directory"));
    }

    // One open beneath SANDBOX_ROOT, then the descriptor only: no window
    // between a check and the open. O_NONBLOCK: a FIFO guest code made must
    // not hold the open (the read below refuses anything but a regular file).
    let fd = open_beneath(
        Path::new(SANDBOX_ROOT),
        below_root(&resolved_path),
        libc::O_RDONLY | libc::O_NOCTTY | libc::O_NONBLOCK,
    )
    .map_err(|e| beneath_error(path, &e, "File not found or inaccessible"))?;
    let file = std::fs::File::from(fd);

    let metadata = file
        .metadata()
        .map_err(|e| CmdError::io(format!("Cannot read '{path}': {e}")))?;

    if metadata.is_dir() {
        return Err(CmdError::path(format!(
            "'{path}' is a directory, not a file"
        )));
    }
    if !metadata.is_file() {
        return Err(CmdError::path(format!("'{path}' is not a regular file")));
    }
    if metadata.len() as usize > MAX_FILE_SIZE_BYTES {
        return Err(CmdError::path(format!(
            "File too large: {} bytes (max {})",
            metadata.len(),
            MAX_FILE_SIZE_BYTES
        )));
    }

    let file_size = metadata.len();

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
            .send(&GuestResponse::FileChunk { data: chunk_b64 })
            .await?;
    }

    writer
        .send(&GuestResponse::FileReadComplete {
            path: path.to_string(),
            size: file_size,
        })
        .await
}

/// Handle list_files command: read directory entries.
pub(crate) async fn handle_list_files(path: &str, writer: &ResponseWriter) -> Result<(), CmdError> {
    let resolved_path = validate_file_path(path)
        .map_err(|e| CmdError::path(format!("Invalid path '{path}': {e}")))?;

    // The folder opened beneath SANDBOX_ROOT, then listed through its
    // descriptor (/proc/self/fd/N names that open folder, not a path again).
    // Entries are not followed: a link reports as a link (not a folder).
    let folder = open_beneath(
        Path::new(SANDBOX_ROOT),
        below_root(&resolved_path),
        libc::O_RDONLY | libc::O_DIRECTORY,
    )
    .map_err(|e| beneath_error(path, &e, "Cannot list"))?;
    let mut read_dir = tokio::fs::read_dir(format!("/proc/self/fd/{}", folder.as_raw_fd()))
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
            // Report actual disk usage, not apparent size. Sparse files
            // (e.g. `truncate("huge", 100GB)`) have large apparent size
            // (metadata.len()) but zero allocated blocks. min(len, blocks*512)
            // preserves correct size for real files while deflating sparse ones.
            size: if metadata.is_dir() {
                0
            } else {
                metadata.len().min(metadata.blocks() * 512)
            },
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));

    writer.send(&GuestResponse::FileList { entries }).await
}

/// Set up a pipelined file write operation.
///
/// Creates the parent folders and a temporary file `.wr.<op_id>.tmp` next to
/// the target, then starts the writer task (`run_write`). Returns the handle
/// of that task, or an error.
pub(crate) fn handle_write_file(
    op_id: &str,
    path: &str,
    make_executable: bool,
) -> Result<ActiveWriteHandle, CmdError> {
    let resolved_path = validate_file_path(path)
        .map_err(|e| CmdError::path(format!("Invalid path '{path}': {e}")))?;

    if resolved_path == std::path::Path::new(SANDBOX_ROOT) {
        return Err(CmdError::path("Cannot write to sandbox root directory"));
    }

    let rel = below_root(&resolved_path);
    let final_name = CString::new(rel.file_name().unwrap_or_default().as_bytes())
        .expect("validate_file_path refuses a NUL byte");
    let tmp_name = CString::new(format!(".wr.{op_id}.tmp"))
        .map_err(|_| CmdError::path(format!("Invalid operation id '{op_id}'")))?;

    // The folders and the temporary file are made through handles opened
    // beneath SANDBOX_ROOT: a link that guest code planted cannot move the
    // write out of it (EXDEV), and O_NOFOLLOW refuses a link at the temporary
    // name. The rename later replaces a link at the final name, never follows it.
    let dir = make_dirs_beneath(
        Path::new(SANDBOX_ROOT),
        rel.parent().unwrap_or(Path::new("")),
    )
    .map_err(|e| beneath_error(path, &e, "Failed to create directories"))?;
    // SAFETY: valid folder descriptor and NUL-terminated name; owned at once.
    let tmp_fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            tmp_name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o600 as libc::c_uint,
        )
    };
    if tmp_fd < 0 {
        let e = std::io::Error::last_os_error();
        return Err(CmdError::io(format!("Failed to create temp file: {e}")));
    }
    // SAFETY: `tmp_fd` is a new descriptor that nothing else owns.
    let file = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(tmp_fd) });

    let counting_writer = CountingWriter {
        inner: file,
        count: 0,
        limit: MAX_FILE_SIZE_BYTES,
    };
    let decoder = match zstd::stream::write::Decoder::new(counting_writer) {
        Ok(d) => d,
        Err(e) => {
            // SAFETY: valid folder descriptor and NUL-terminated name.
            unsafe { libc::unlinkat(dir.as_raw_fd(), tmp_name.as_ptr(), 0) };
            return Err(CmdError::io(format!("Decompression init failed: {e}")));
        }
    };

    let (chunk_tx, chunk_rx) = mpsc::channel::<Vec<u8>>(16);
    let write_state = WriteFileState {
        decoder,
        dir,
        tmp_name,
        final_name,
        make_executable,
        path_display: path.to_string(),
        finished: false,
    };

    let end = Arc::new(WriteEnd::new());
    let task_end = Arc::clone(&end);
    let task = tokio::task::spawn_blocking(move || run_write(write_state, chunk_rx, &task_end));

    Ok(ActiveWriteHandle {
        chunk_tx,
        task,
        path_display: path.to_string(),
        end,
    })
}

/// Body of the writer task of a pipelined write (a blocking thread).
///
/// Decompresses each chunk into the temporary file. An empty chunk is the
/// end marker: flush, set the mode and owner, claim the commit (`WriteEnd`),
/// then rename into place. A closed channel without the end marker, or an
/// abandon that the agent claimed first, ends the write; dropping `state` then
/// removes the temporary file. Returns the bytes written, or the message of
/// an `io_error` reply.
fn run_write(
    mut state: WriteFileState,
    mut chunk_rx: mpsc::Receiver<Vec<u8>>,
    end: &WriteEnd,
) -> Result<usize, String> {
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
    // Mode and owner through the open file (fchmod, fchown): no path to swap.
    let mode = if state.make_executable { 0o755 } else { 0o644 };
    let file = &state.decoder.get_ref().inner;
    if let Err(e) = file.set_permissions(std::fs::Permissions::from_mode(mode)) {
        return Err(state.io_error(format!("set_permissions: {e}")));
    }
    if let Err(e) = std::os::unix::fs::fchown(file, Some(SANDBOX_UID), Some(SANDBOX_GID)) {
        return Err(state.io_error(format!("chown: {e}")));
    }
    // The agent reports an error for an abandoned write: never commit one
    if !end.try_commit() {
        return Err(state.io_error("abandoned after a stall"));
    }
    // SAFETY: valid folder descriptor and NUL-terminated names. renameat
    // replaces a link at the final name; it never follows it.
    let renamed = unsafe {
        libc::renameat(
            state.dir.as_raw_fd(),
            state.tmp_name.as_ptr(),
            state.dir.as_raw_fd(),
            state.final_name.as_ptr(),
        )
    };
    if renamed != 0 {
        let e = std::io::Error::last_os_error();
        return Err(state.io_error(format!("rename: {e}")));
    }
    state.finished = true;
    Ok(bytes_written)
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

    // The host's traversal vectors (tests/test_file_io_security.py), checked here where the
    // rule lives (a lexical check, no file is opened): each is refused
    #[test]
    fn test_validate_file_path_refuses_the_traversal_vectors() {
        for path in [
            "../etc/passwd",
            "../../etc/shadow",
            "../../../root/.ssh/id_rsa",
            "subdir/../../etc/passwd",
            "subdir/../subdir/../../etc/passwd",
            "/etc/passwd",
            "/home/user/../../../etc/passwd",
            "../etc",
            "../../",
            "/etc",
            "subdir/../../etc",
        ] {
            assert!(validate_file_path(path).is_err(), "{path}");
        }
    }

    // Web encodings (fuzzdb) are literal names on Unix: a path under the sandbox root
    #[test]
    fn test_validate_file_path_web_encodings_are_literal_names() {
        for (path, real) in [
            ("....//....//etc/passwd", "/home/user/..../..../etc/passwd"),
            ("..%2f..%2fetc/passwd", "/home/user/..%2f..%2fetc/passwd"),
            ("..\\..\\etc\\passwd", "/home/user/..\\..\\etc\\passwd"),
        ] {
            assert_eq!(
                validate_file_path(path).unwrap(),
                std::path::PathBuf::from(real),
                "{path}"
            );
        }
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

    /// A relative path of `len` bytes: components of 254 bytes (below NAME_MAX), then the rest.
    fn path_of_len(len: usize) -> String {
        let mut path = String::new();
        while path.len() < len {
            if !path.is_empty() {
                path.push('/');
            }
            let take = (len - path.len()).min(254);
            path.push_str(&"a".repeat(take));
        }
        path
    }

    #[test]
    fn test_validate_file_path_exactly_max_length() {
        let path = path_of_len(MAX_FILE_PATH_LENGTH);
        assert_eq!(path.len(), 4096);
        assert!(validate_file_path(&path).is_ok());
    }

    #[test]
    fn test_validate_file_path_one_byte_over_max_length() {
        let path = path_of_len(MAX_FILE_PATH_LENGTH + 1);
        assert_eq!(path.len(), 4097);
        assert!(validate_file_path(&path).is_err());
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
    // run_write: the commit point
    // -------------------------------------------------------------------------

    /// How the agent ends a write of `write_through_pipeline`.
    enum Ending {
        /// The end marker (file_end).
        EndMarker,
        /// The end marker, after the agent abandoned the write at a stall.
        Abandoned,
        /// No end marker: the channel closes (the host closed its connection).
        HostGone,
    }

    /// Run a whole write of `data` into a fresh temp folder.
    fn write_through_pipeline(
        name: &str,
        data: &[u8],
        ending: Ending,
    ) -> (Result<usize, String>, std::path::PathBuf) {
        let dir = std::fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("guest-agent-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = std::fs::File::create(dir.join(".wr.op.tmp")).unwrap();
        let state = WriteFileState {
            decoder: zstd::stream::write::Decoder::new(CountingWriter {
                inner: file,
                count: 0,
                limit: MAX_FILE_SIZE_BYTES,
            })
            .unwrap(),
            dir: open_beneath(&dir, Path::new(""), libc::O_PATH | libc::O_DIRECTORY).unwrap(),
            tmp_name: CString::new(".wr.op.tmp").unwrap(),
            final_name: CString::new("file.bin").unwrap(),
            make_executable: false,
            path_display: "file.bin".to_string(),
            finished: false,
        };
        let (chunk_tx, chunk_rx) = mpsc::channel::<Vec<u8>>(16);
        let compressed = zstd::stream::encode_all(data, FILE_TRANSFER_ZSTD_LEVEL).unwrap();
        chunk_tx.try_send(compressed).unwrap();
        let end = WriteEnd::new();
        match ending {
            Ending::EndMarker => chunk_tx.try_send(vec![]).unwrap(),
            Ending::Abandoned => {
                chunk_tx.try_send(vec![]).unwrap();
                assert!(end.try_abandon());
            }
            Ending::HostGone => drop(chunk_tx),
        }
        (run_write(state, chunk_rx, &end), dir)
    }

    #[test]
    fn test_run_write_commits_a_finished_write() {
        let (result, dir) = write_through_pipeline("commit", b"hello", Ending::EndMarker);
        assert_eq!(result.ok().unwrap(), 5);
        assert_eq!(std::fs::read(dir.join("file.bin")).unwrap(), b"hello");
        assert!(!dir.join(".wr.op.tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_run_write_never_commits_an_abandoned_write() {
        let (result, dir) = write_through_pipeline("abandoned", b"hello", Ending::Abandoned);
        assert_eq!(
            result.unwrap_err(),
            "I/O error for 'file.bin': abandoned after a stall"
        );
        assert!(
            !dir.join("file.bin").exists(),
            "an abandoned write must not land"
        );
        assert!(
            !dir.join(".wr.op.tmp").exists(),
            "its temporary file is removed"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_run_write_drops_a_write_whose_host_is_gone() {
        // The host closed its connection before file_end: the agent drops the
        // write's channel. No file lands, and the temporary file is removed.
        let (result, dir) = write_through_pipeline("host-gone", b"hello", Ending::HostGone);
        assert_eq!(
            result.unwrap_err(),
            "I/O error for 'file.bin': channel closed without finalize"
        );
        assert!(!dir.join("file.bin").exists());
        assert!(!dir.join(".wr.op.tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // -------------------------------------------------------------------------
    // Resolution beneath the root
    // -------------------------------------------------------------------------

    /// A fresh `root` folder, a folder `outside` next to it, and links in `root`
    /// that guest code could plant: `out` (absolute, out of root), `rel_out`
    /// (relative, out of root), `in` (relative, to root/inside), `abs_in`
    /// (absolute, to root/inside), `loop` (to itself), `fd` (a /proc magic
    /// link) and `dangling` (absolute, to outside/new, which does not exist).
    fn root_with_links(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let base = std::fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!("guest-agent-beneath-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("root");
        let outside = base.join("outside");
        std::fs::create_dir_all(root.join("inside")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret"), b"x").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("out")).unwrap();
        std::os::unix::fs::symlink("../outside", root.join("rel_out")).unwrap();
        std::os::unix::fs::symlink("inside", root.join("in")).unwrap();
        std::os::unix::fs::symlink(root.join("inside"), root.join("abs_in")).unwrap();
        std::os::unix::fs::symlink("loop", root.join("loop")).unwrap();
        std::os::unix::fs::symlink("/proc/self/fd/0", root.join("fd")).unwrap();
        std::os::unix::fs::symlink(outside.join("new"), root.join("dangling")).unwrap();
        (base, root)
    }

    #[test]
    fn test_open_beneath_refuses_links_out_of_the_root() {
        let (base, root) = root_with_links("open");
        for rel in ["out/secret", "rel_out/secret", "../outside/secret"] {
            let e = open_beneath(&root, Path::new(rel), libc::O_RDONLY).unwrap_err();
            assert_eq!(e.raw_os_error(), Some(libc::EXDEV), "{rel}");
        }
        for rel in ["loop", "fd"] {
            let e = open_beneath(&root, Path::new(rel), libc::O_RDONLY).unwrap_err();
            assert_eq!(e.raw_os_error(), Some(libc::ELOOP), "{rel}");
        }
        std::fs::write(root.join("inside").join("f"), b"ok").unwrap();
        for rel in ["in/f", "abs_in/f"] {
            let file =
                std::fs::File::from(open_beneath(&root, Path::new(rel), libc::O_RDONLY).unwrap());
            assert_eq!(std::io::read_to_string(file).unwrap(), "ok", "{rel}");
        }
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn test_make_dirs_beneath_makes_nothing_out_of_the_root() {
        let (base, root) = root_with_links("dirs");
        for rel in ["out/new", "rel_out/new"] {
            let e = make_dirs_beneath(&root, Path::new(rel)).unwrap_err();
            assert_eq!(e.raw_os_error(), Some(libc::EXDEV), "{rel}");
        }
        // A link to a missing folder out of root: mkdirat meets the link (EEXIST)
        // and never makes its target (a guard: any mkdir-based walk holds it)
        assert!(make_dirs_beneath(&root, Path::new("dangling/x")).is_err());
        assert!(!base.join("outside").join("new").exists());
        assert!(make_dirs_beneath(&root, Path::new("a/b")).is_ok());
        assert!(root.join("a").join("b").is_dir());
        // The folders it makes belong to the sandbox user; one that was there keeps its owner
        for made in [root.join("a"), root.join("a").join("b")] {
            assert_eq!(
                std::fs::metadata(&made).unwrap().uid(),
                SANDBOX_UID,
                "{made:?}"
            );
        }
        let inside_owner = std::fs::metadata(root.join("inside")).unwrap().uid();
        assert!(make_dirs_beneath(&root, Path::new("in/c")).is_ok());
        assert!(root.join("inside").join("c").is_dir());
        assert!(make_dirs_beneath(&root, Path::new("abs_in/d")).is_ok());
        assert!(root.join("inside").join("d").is_dir());
        assert_eq!(
            std::fs::metadata(root.join("inside")).unwrap().uid(),
            inside_owner
        );
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn test_open_beneath_opens_nothing_out_of_the_root() {
        // A FIFO out of root and a link to it in root. A writer that opens the
        // FIFO waits for a reader: an open that reached the FIFO through the
        // link (not only its O_PATH handle) would release it.
        let (base, root) = root_with_links("fifo");
        let fifo = base.join("outside").join("fifo");
        let c_fifo = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        // SAFETY: a NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(c_fifo.as_ptr(), 0o600) }, 0);
        std::os::unix::fs::symlink(&fifo, root.join("fifo")).unwrap();
        let (opened_tx, opened_rx) = std::sync::mpsc::channel::<()>();
        let writer_fifo = fifo.clone();
        let writer = std::thread::spawn(move || {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&writer_fifo)
                .unwrap();
            opened_tx.send(()).unwrap();
            file
        });
        std::thread::sleep(std::time::Duration::from_millis(50)); // the writer waits in its open

        let e =
            open_beneath(&root, Path::new("fifo"), libc::O_RDONLY | libc::O_NONBLOCK).unwrap_err();

        assert_eq!(e.raw_os_error(), Some(libc::EXDEV));
        assert!(
            opened_rx
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "no reader may have opened the FIFO out of root"
        );
        let _reader = std::fs::File::open(&fifo).unwrap(); // release the writer
        drop(writer.join().unwrap());
        std::fs::remove_dir_all(&base).unwrap();
    }
}

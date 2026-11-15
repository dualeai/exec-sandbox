//! Connection handling: non-blocking virtio-serial I/O and command dispatch.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tokio::io::unix::AsyncFd;
use tokio::io::{AsyncWriteExt, BufWriter, Interest};
use tokio::sync::mpsc;

use crate::constants::*;
use crate::error::*;
use crate::file_io::{
    ActiveWriteHandle, WriteEnd, handle_list_files, handle_read_file, handle_write_file,
};
use crate::packages::install_packages;
use crate::repl::execute::{execute_code_streaming, warm_repl};
use crate::types::{GuestCommand, GuestResponse};

/// Extract `op_id` from raw JSON before serde deserialization.
///
/// Only the file-transfer variants of `GuestCommand` have an `op_id` field;
/// the `ResponseWriter` needs it for the replies to every command.
fn extract_op_id(value: &serde_json::Value) -> Option<String> {
    value
        .get("op_id")
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// Host wall clock at send time (top-level, like op_id: not part of GuestCommand).
fn extract_host_time_ns(value: &serde_json::Value) -> Option<u64> {
    value
        .get("host_time_ns")
        .and_then(serde_json::Value::as_u64)
}

// ============================================================================
// CommandPort
// ============================================================================

/// Reader of the host → guest command port (virtio-serial).
///
/// The port stays open while it works. When the guest closes a virtio-serial
/// port, the guest kernel drops the bytes it holds and the bytes QEMU still
/// sends before QEMU sees the close (virtio_console `port_fops_release` and
/// `in_intr` call `discard_port_data`). A host request that arrives then is
/// lost, and the host waits for its full deadline. So the agent never closes
/// the port to wait for a host: when the host closes its socket, a read gives
/// 0 bytes, and the bytes of the next host arrive on the same descriptor.
///
/// Readiness comes from a private edge-triggered epoll set that holds the
/// port, and tokio watches that set. tokio's own registration of the port
/// would keep the HUP of a closed host as a final state, so each wait after
/// the first host close would return at once.
///
/// All bytes read but not yet returned stay in `pending`, so a cancelled
/// `read_line` loses nothing.
pub(crate) struct CommandPort {
    port: std::fs::File,
    poller: AsyncFd<OwnedFd>,
    pending: Vec<u8>,
    /// Bytes at the start of `pending` already searched for a newline.
    scanned: usize,
    /// True from a reported host close (or the start) until the next byte.
    host_gone: bool,
}

impl CommandPort {
    /// Open the port for non-blocking reads and add it to a private
    /// edge-triggered epoll set. No host needs to be connected.
    pub(crate) fn open(path: &str) -> std::io::Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;

        let port = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)?;
        // SAFETY: a plain syscall; the result is checked before use.
        let raw = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
        if raw < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: `raw` is a new descriptor that nothing else owns.
        let poller = unsafe { OwnedFd::from_raw_fd(raw) };
        let mut event = libc::epoll_event {
            events: (libc::EPOLLIN | libc::EPOLLET) as u32,
            u64: 0,
        };
        // SAFETY: both descriptors are open and `event` outlives the call.
        let added = unsafe {
            libc::epoll_ctl(
                poller.as_raw_fd(),
                libc::EPOLL_CTL_ADD,
                port.as_raw_fd(),
                &mut event,
            )
        };
        if added < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: AsyncFd owns `poller`, so the fd stays open and unchanged for its lifetime.
        let poller = unsafe { AsyncFd::register_with_interest(poller, Interest::READABLE) }?;
        Ok(Self {
            port,
            poller,
            pending: Vec::new(),
            scanned: 0,
            host_gone: true,
        })
    }

    /// Append the next line (with its newline) to `buf` and return its length.
    ///
    /// Returns 0 once when the host closes its side. The port stays open, and
    /// the next call waits for the bytes of the next host. Bytes of a line
    /// that is not complete stay in `pending` across the host close.
    pub(crate) async fn read_line(&mut self, buf: &mut String) -> std::io::Result<usize> {
        loop {
            if let Some(pos) = self.pending[self.scanned..]
                .iter()
                .position(|&b| b == b'\n')
            {
                let line_len = self.scanned + pos + 1;
                buf.push_str(&String::from_utf8_lossy(&self.pending[..line_len]));
                self.pending.drain(..line_len);
                self.scanned = 0;
                return Ok(line_len);
            }
            self.scanned = self.pending.len();

            // Edge-triggered: wait only after a read that found no bytes.
            match self.read_some() {
                Ok(0) if self.host_gone => self.wait().await?,
                Ok(0) => {
                    self.host_gone = true;
                    return Ok(0);
                }
                Ok(_) => self.host_gone = false,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => self.wait().await?,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }

    /// One non-blocking read, appended to `pending`. 0 means that no host is
    /// connected and that the kernel holds no byte for the port.
    fn read_some(&mut self) -> std::io::Result<usize> {
        let mut chunk = [0u8; 16384];
        // SAFETY: `chunk` is valid for writes of its full length.
        let n = unsafe {
            libc::read(
                self.port.as_raw_fd(),
                chunk.as_mut_ptr().cast::<libc::c_void>(),
                chunk.len(),
            )
        };
        if n < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let n = n as usize;
        self.pending.extend_from_slice(&chunk[..n]);
        Ok(n)
    }

    /// Wait for the next wake of the port: new bytes, or a host close.
    async fn wait(&self) -> std::io::Result<()> {
        let mut guard = self.poller.readable().await?;
        // Take the reported edge off the private set. The caller reads next,
        // so a wake that comes after this call is never missed.
        let mut events = [libc::epoll_event { events: 0, u64: 0 }];
        // SAFETY: `events` is valid for one entry; timeout 0 never blocks.
        let n = unsafe { libc::epoll_wait(self.poller.as_raw_fd(), events.as_mut_ptr(), 1, 0) };
        if n < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
        guard.clear_ready();
        Ok(())
    }
}

// ============================================================================
// Connection handler
// ============================================================================

/// Bound on each wait for a pipelined file write.
const WRITE_STALL_TIMEOUT: Duration = Duration::from_secs(FILE_WRITE_STALL_TIMEOUT_SECONDS);

/// How a pipelined write ended, as `finish_write` saw it.
enum WriteFinish {
    /// The writer task ended: its result (bytes written, or an error message).
    Done(Result<Result<usize, String>, tokio::task::JoinError>),
    /// Abandoned at the limit: the task never renames its temporary file, so
    /// an error reply stays true.
    Abandoned,
    /// The task claimed the commit (`WriteEnd`), but its rename did not end
    /// within `commit_limit` (a stalled disk): the write may still land, so the
    /// agent sends no reply.
    CommitStalled,
}

/// Send the end marker of a pipelined write and wait for its writer task, at
/// most `limit`. Past it, the write is abandoned, unless the task claimed the
/// commit first: then its rename is on its way, and its result is awaited at
/// most `limit` more, so a stalled disk never stops the agent, which reads one
/// command at a time.
async fn finish_write(
    chunk_tx: mpsc::Sender<Vec<u8>>,
    mut task: tokio::task::JoinHandle<Result<usize, String>>,
    end: &WriteEnd,
    limit: Duration,
) -> WriteFinish {
    let finish = async {
        let _ = chunk_tx.send(vec![]).await;
        drop(chunk_tx);
        (&mut task).await
    };
    match tokio::time::timeout(limit, finish).await {
        Ok(finished) => WriteFinish::Done(finished),
        Err(_) if end.try_abandon() => WriteFinish::Abandoned,
        Err(_) => match tokio::time::timeout(limit, task).await {
            Ok(finished) => WriteFinish::Done(finished),
            Err(_) => WriteFinish::CommitStalled,
        },
    }
}

/// The io_error message of a write whose task ended without its file
/// (`ended`, logged): the task's message, or its join error.
fn write_error(
    ended: Result<Result<usize, String>, tokio::task::JoinError>,
    path_display: &str,
    op_id: &str,
) -> String {
    let message = match ended {
        Ok(Err(message)) => message,
        Ok(Ok(_)) => format!("Write pipeline unexpectedly closed for '{path_display}'"),
        Err(join_err) => format!("Internal error writing '{path_display}': {join_err}"),
    };
    log_error!("Write pipeline error for '{path_display}' (op_id={op_id}): {message}");
    message
}

/// Report a command-handler outcome on the dispatch loop.
///
/// `Ok(())` and `Reply` errors are recoverable (the loop continues); only a
/// closed write queue or a `Fatal` error terminate the connection. Used by
/// install, execute, read and list:
/// `if let Err(e) = report(&writer, handler(...).await).await { break Err(e); }`.
async fn report(
    writer: &ResponseWriter,
    outcome: Result<(), CmdError>,
) -> Result<(), Box<dyn std::error::Error>> {
    match outcome {
        Ok(()) => Ok(()),
        Err(CmdError::Reply {
            message,
            error_type,
        }) => writer
            .send_error(message, error_type)
            .await
            .map_err(|_| "write queue closed".into()),
        Err(CmdError::Fatal(e)) => Err(e.to_string().into()),
    }
}

/// Answer a Ping: a Pong. With host entropy, `mix` (entropy::mix_host_entropy) adds it to
/// the kernel and reseeds first; when that fails, an error and no Pong.
async fn answer_ping(
    writer: &ResponseWriter,
    host_entropy: Option<&str>,
    mix: impl FnOnce(&str) -> Result<(), String>,
) -> Result<(), CmdError> {
    if let Some(host_entropy) = host_entropy
        && let Err(error) = mix(host_entropy)
    {
        let message = format!("kernel random reseed with the host entropy failed: {error}");
        log_error!("{message}");
        return writer.send_error(message, ErrorType::Io.as_str()).await;
    }
    writer.send(&GuestResponse::Pong { version: VERSION }).await
}

/// Serve one host connection: from the first byte to the host close.
///
/// Returns `Ok(())` when the host closes its side. It drops the writes that
/// did not get their `file_end`, as the host expects on a close. The caller
/// keeps both ports open for the next host. There is no idle timeout: an
/// idle host is not a fault, and only a host close ends the connection.
pub(crate) async fn handle_connection(
    cmd_reader: &mut CommandPort,
    event_file: tokio::fs::File,
) -> Result<(), Box<dyn std::error::Error>> {
    let writer = BufWriter::new(event_file);

    let (write_tx, mut write_rx) = mpsc::channel::<Vec<u8>>(WRITE_QUEUE_SIZE);

    // Spawn write task
    let write_handle = tokio::spawn(async move {
        let mut writer = writer;
        let mut pending = 0u32;
        while let Some(data) = write_rx.recv().await {
            if let Err(e) = writer.write_all(&data).await {
                log_error!("Write error: {e}");
                break;
            }
            pending += 1;
            if pending >= 16 || write_rx.is_empty() {
                if let Err(e) = writer.flush().await {
                    log_error!("Flush error: {e}");
                    break;
                }
                pending = 0;
            }
        }
        let _ = writer.flush().await;
    });

    let mut active_writes: HashMap<String, ActiveWriteHandle> = HashMap::new();

    let mut line = String::new();
    let result = loop {
        line.clear();

        let bytes_read = match cmd_reader.read_line(&mut line).await {
            Ok(0) => {
                log_info!("Connection closed by client");
                break Ok(());
            }
            Ok(n) => n,
            Err(e) => {
                log_error!("Read error: {e}");
                break Err(e.into());
            }
        };

        if bytes_read > MAX_REQUEST_SIZE_BYTES {
            let _ = ResponseWriter::new(write_tx.clone(), None)
                .send_error(
                    format!(
                        "Request too large: {} bytes (max {} bytes)",
                        bytes_read, MAX_REQUEST_SIZE_BYTES
                    ),
                    ErrorType::Request.as_str(),
                )
                .await;
            continue;
        }

        log_info!("Received request ({bytes_read} bytes)");

        // Two-phase parse: extract op_id and host_time_ns from the raw JSON,
        // then deserialize the command. Most GuestCommand variants have no
        // op_id field; the ResponseWriter needs it for every reply.
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                log_error!("JSON parse error: {e}");
                let _ = ResponseWriter::new(write_tx.clone(), None)
                    .send_error(format!("Invalid JSON: {e}"), ErrorType::Request.as_str())
                    .await;
                continue;
            }
        };
        let op_id = extract_op_id(&value);
        let host_time_ns = extract_host_time_ns(&value);
        let cmd: GuestCommand = match serde_json::from_value(value) {
            Ok(cmd) => cmd,
            Err(e) => {
                log_error!("Command parse error: {e}");
                let _ = ResponseWriter::new(write_tx.clone(), op_id)
                    .send_error(format!("Invalid command: {e}"), ErrorType::Request.as_str())
                    .await;
                continue;
            }
        };

        // Step the wall clock forward to the host's before any user code reads it.
        crate::clock::sync(host_time_ns);

        // Reseed the kernel CRNG before each command (~11 µs; a failure only logs,
        // entropy.rs), so no user code reads the CRNG with the key it had before this
        // command. A file chunk is never first: its write_file reseeded already.
        if !matches!(cmd, GuestCommand::FileChunk { .. }) {
            crate::entropy::reseed();
        }

        let writer = ResponseWriter::new(write_tx.clone(), op_id);

        match cmd {
            GuestCommand::Ping { host_entropy } => {
                log_info!("Processing: ping (host_entropy={})", host_entropy.is_some());
                if answer_ping(
                    &writer,
                    host_entropy.as_deref(),
                    crate::entropy::mix_host_entropy,
                )
                .await
                .is_err()
                {
                    break Err("write queue closed".into());
                }
            }
            GuestCommand::WarmRepl { language } => {
                log_info!("Processing: warm_repl (language={})", language.as_str());
                let warmed = match crate::init::wait_for_untrusted_readiness().await {
                    Ok(()) => warm_repl(language).await,
                    Err(error) => {
                        log_warn!("REPL warm-up readiness failed: {error}");
                        Err(format!("guest readiness failed: {error}"))
                    }
                };
                let (status, message) = match warmed {
                    Ok(()) => ("ok", None),
                    Err(message) => ("error", Some(message)),
                };
                if writer
                    .send(&GuestResponse::WarmReplAck {
                        status: status.to_string(),
                        message,
                    })
                    .await
                    .is_err()
                {
                    break Err("write queue closed".into());
                }
            }
            GuestCommand::InstallPackages {
                language,
                packages,
                timeout,
            } => {
                // Gate all process spawning on network and capped-zram safety.
                if let Err(error) = crate::init::wait_for_untrusted_readiness().await {
                    if writer
                        .send_error(
                            format!("VM readiness failed before package installation: {error}"),
                            ErrorType::Execution.as_str(),
                        )
                        .await
                        .is_err()
                    {
                        break Err("write queue closed".into());
                    }
                    continue;
                }
                log_info!(
                    "Processing: install_packages (language={}, count={}, timeout={timeout}s)",
                    language.as_str(),
                    packages.len()
                );
                if let Err(e) = report(
                    &writer,
                    install_packages(language, &packages, timeout, &writer).await,
                )
                .await
                {
                    break Err(e);
                }
            }
            GuestCommand::ExecuteCode {
                language,
                code,
                timeout,
                env_vars,
                reseed,
            } => {
                // Gate all process spawning on network and capped-zram safety.
                let t_net = crate::monotonic_ms();
                if let Err(error) = crate::init::wait_for_untrusted_readiness().await {
                    if writer
                        .send_error(
                            format!("VM readiness failed before code execution: {error}"),
                            ErrorType::Execution.as_str(),
                        )
                        .await
                        .is_err()
                    {
                        break Err("write queue closed".into());
                    }
                    continue;
                }
                let t_net_done = crate::monotonic_ms();
                log_info!(
                    "[timing] wait_for_untrusted_readiness: {}ms (at {}ms)",
                    t_net_done - t_net,
                    t_net_done
                );
                log_info!(
                    "Processing: execute_code (language={}, code_size={}, timeout={timeout}s, env_vars={}, reseed={reseed})",
                    language.as_str(),
                    code.len(),
                    env_vars.len()
                );
                let outcome =
                    execute_code_streaming(language, &code, timeout, &env_vars, reseed, &writer)
                        .await;
                if let Err(e) = report(&writer, outcome).await {
                    break Err(e);
                }
            }
            GuestCommand::WriteFile {
                op_id,
                path,
                make_executable,
            } => {
                log_info!(
                    "Processing: write_file (op_id={op_id}, path={path}, executable={make_executable})"
                );
                match active_writes.entry(op_id) {
                    Entry::Occupied(entry) => {
                        let _ = writer
                            .send_error(
                                format!("Duplicate op_id '{}' for write_file", entry.key()),
                                ErrorType::Protocol.as_str(),
                            )
                            .await;
                    }
                    Entry::Vacant(slot) => {
                        match handle_write_file(slot.key(), &path, make_executable) {
                            Ok(handle) => {
                                slot.insert(handle);
                            }
                            Err(CmdError::Reply {
                                message,
                                error_type,
                            }) => {
                                let _ = writer.send_error(message, error_type).await;
                            }
                            Err(CmdError::Fatal(e)) => break Err(e.to_string().into()),
                        }
                    }
                }
            }
            GuestCommand::FileChunk { op_id, data } => {
                let Some(handle) = active_writes.get(&op_id) else {
                    let _ = writer
                        .send_error(
                            format!("No active write for op_id '{op_id}'"),
                            ErrorType::Protocol.as_str(),
                        )
                        .await;
                    continue;
                };
                let decoded = match BASE64.decode(&data) {
                    Ok(d) => d,
                    Err(e) => {
                        let message = format!(
                            "Invalid base64 in chunk for '{}': {}",
                            handle.path_display, e
                        );
                        // Its chunk_tx goes: the task ends, and removes its temporary file
                        active_writes.remove(&op_id);
                        let _ = writer
                            .send_error(message, ErrorType::Protocol.as_str())
                            .await;
                        continue;
                    }
                };
                // An empty chunk holds no data, and an empty vector is the end
                // marker that only finish_write sends to the task
                if decoded.is_empty() {
                    continue;
                }
                // Bounded: a writer stuck on the disk must not stop the agent,
                // which reads one command at a time.
                let sent =
                    tokio::time::timeout(WRITE_STALL_TIMEOUT, handle.chunk_tx.send(decoded)).await;
                let Ok(sent) = sent else {
                    // No end marker went to the task: it never commits, and
                    // the drop of its chunk_tx ends it
                    let handle = active_writes.remove(&op_id).unwrap();
                    log_error!(
                        "Write of '{}' (op_id={op_id}) took no chunk in {FILE_WRITE_STALL_TIMEOUT_SECONDS}s, abandoned",
                        handle.path_display
                    );
                    let _ = writer
                        .send_error(
                            format!(
                                "Write of '{}' stalled: no chunk written in {FILE_WRITE_STALL_TIMEOUT_SECONDS}s",
                                handle.path_display
                            ),
                            ErrorType::Io.as_str(),
                        )
                        .await;
                    continue;
                };
                if sent.is_err() {
                    // The writer task ended early (an I/O error): it holds the reason.
                    let handle = active_writes.remove(&op_id).unwrap();
                    drop(handle.chunk_tx);
                    // Bounded: its end removes the temporary file, which a stalled disk
                    // can hold; the write failed either way (no end marker)
                    let message = match tokio::time::timeout(WRITE_STALL_TIMEOUT, handle.task).await
                    {
                        // With no end marker, the task ends with an error only
                        Ok(ended) => write_error(ended, &handle.path_display, &op_id),
                        Err(_) => {
                            log_error!(
                                "Write of '{}' (op_id={op_id}) failed, and its cleanup did not end in {FILE_WRITE_STALL_TIMEOUT_SECONDS}s",
                                handle.path_display
                            );
                            format!("Write of '{}' failed", handle.path_display)
                        }
                    };
                    let _ = writer.send_error(message, ErrorType::Io.as_str()).await;
                }
            }
            GuestCommand::FileEnd { op_id } => {
                let ActiveWriteHandle {
                    chunk_tx,
                    task,
                    path_display,
                    end,
                } = match active_writes.remove(&op_id) {
                    Some(h) => h,
                    None => {
                        let _ = writer
                            .send_error(
                                format!("No active write for op_id '{op_id}'"),
                                ErrorType::Protocol.as_str(),
                            )
                            .await;
                        continue;
                    }
                };
                let finished = match finish_write(chunk_tx, task, &end, WRITE_STALL_TIMEOUT).await {
                    WriteFinish::Done(finished) => finished,
                    WriteFinish::Abandoned => {
                        log_error!(
                            "Write of '{path_display}' (op_id={op_id}) did not finish in {FILE_WRITE_STALL_TIMEOUT_SECONDS}s, abandoned"
                        );
                        let _ = writer
                            .send_error(
                                format!(
                                    "Write of '{path_display}' stalled: not finished in {FILE_WRITE_STALL_TIMEOUT_SECONDS}s"
                                ),
                                ErrorType::Io.as_str(),
                            )
                            .await;
                        continue;
                    }
                    WriteFinish::CommitStalled => {
                        // No reply: no error type tells the host "it may still land", and
                        // an io_error says it failed. The host's deadline gives an unknown
                        // outcome (OutcomeUnknownError), which is the truth
                        log_error!(
                            "Write of '{path_display}' (op_id={op_id}) stalled in its rename; it may still land, no reply"
                        );
                        continue;
                    }
                };
                if let Ok(Ok(bytes_written)) = finished {
                    if writer
                        .send(&GuestResponse::FileWriteAck {
                            path: path_display,
                            bytes_written,
                        })
                        .await
                        .is_err()
                    {
                        break Err("write queue closed".into());
                    }
                } else {
                    let message = write_error(finished, &path_display, &op_id);
                    let _ = writer.send_error(message, ErrorType::Io.as_str()).await;
                }
            }
            GuestCommand::ReadFile { op_id, path } => {
                log_info!("Processing: read_file (op_id={op_id}, path={path})");
                if let Err(e) = report(&writer, handle_read_file(&path, &writer).await).await {
                    break Err(e);
                }
            }
            GuestCommand::ListFiles { path } => {
                log_info!("Processing: list_files (path={path})");
                if let Err(e) = report(&writer, handle_list_files(&path, &writer).await).await {
                    break Err(e);
                }
            }
        }
    };

    drop(write_tx);
    let _ = write_handle.await;
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    // -------------------------------------------------------------------------
    // CommandPort
    //
    // A FIFO acts like the virtio-serial port: with no writer, a read gives 0
    // bytes (no host); a new writer brings bytes again on the same descriptor
    // (the next host), and its open wakes the reader's wait queue.
    // -------------------------------------------------------------------------

    use std::io::Write as _;

    /// Number of the next FIFO folder of this test process.
    static NEXT_FIFO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// A FIFO in a fresh temp folder, removed on drop.
    struct Fifo {
        dir: std::path::PathBuf,
        path: std::path::PathBuf,
    }

    impl Fifo {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "guest-agent-{name}-{}-{}",
                std::process::id(),
                NEXT_FIFO.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("port");
            let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            // SAFETY: a valid NUL-terminated path.
            assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
            Self { dir, path }
        }

        fn reader(&self) -> CommandPort {
            CommandPort::open(self.path.to_str().unwrap()).unwrap()
        }

        /// A host: opens the FIFO for writing (the reader is already open).
        fn host(&self) -> std::fs::File {
            std::fs::OpenOptions::new()
                .write(true)
                .open(&self.path)
                .unwrap()
        }
    }

    impl Drop for Fifo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    async fn line_of(port: &mut CommandPort) -> String {
        let mut line = String::new();
        let n = tokio::time::timeout(Duration::from_secs(2), port.read_line(&mut line))
            .await
            .expect("read_line must not hang")
            .unwrap();
        assert_eq!(n, line.len());
        line
    }

    async fn host_close_of(port: &mut CommandPort) {
        let mut line = String::new();
        let n = tokio::time::timeout(Duration::from_secs(2), port.read_line(&mut line))
            .await
            .expect("read_line must report the host close")
            .unwrap();
        assert_eq!((n, line.as_str()), (0, ""));
    }

    #[tokio::test]
    async fn port_reads_the_next_host_after_a_host_close() {
        let fifo = Fifo::new("next-host");
        let mut port = fifo.reader();

        let mut first = fifo.host();
        first.write_all(b"{\"a\":1}\n").unwrap();
        assert_eq!(line_of(&mut port).await, "{\"a\":1}\n");
        drop(first);
        host_close_of(&mut port).await;

        let mut second = fifo.host();
        second.write_all(b"{\"b\":2}\n").unwrap();
        assert_eq!(line_of(&mut port).await, "{\"b\":2}\n");
    }

    #[tokio::test]
    async fn port_waits_after_a_host_close_instead_of_reporting_it_again() {
        let fifo = Fifo::new("one-close");
        let mut port = fifo.reader();
        let mut first = fifo.host();
        first.write_all(b"ping\n").unwrap();
        assert_eq!(line_of(&mut port).await, "ping\n");
        drop(first);
        host_close_of(&mut port).await;

        // No host: the next read waits; it does not report the close again.
        let mut line = String::new();
        let waited =
            tokio::time::timeout(Duration::from_millis(50), port.read_line(&mut line)).await;
        assert!(waited.is_err(), "got {waited:?} with no host");

        let mut host = fifo.host();
        host.write_all(b"ping\n").unwrap();
        assert_eq!(line_of(&mut port).await, "ping\n");
    }

    #[tokio::test]
    async fn port_keeps_a_partial_line_across_a_host_close() {
        let fifo = Fifo::new("partial-close");
        let mut port = fifo.reader();

        let mut first = fifo.host();
        first.write_all(b"{\"par").unwrap();
        drop(first);
        host_close_of(&mut port).await;

        let mut second = fifo.host();
        second.write_all(b"tial\":1}\n").unwrap();
        assert_eq!(line_of(&mut port).await, "{\"partial\":1}\n");
    }

    #[tokio::test]
    async fn port_keeps_the_bytes_of_a_cancelled_read() {
        let fifo = Fifo::new("cancelled");
        let mut port = fifo.reader();
        let mut host = fifo.host();

        host.write_all(b"{\"op\":").unwrap();
        let mut line = String::new();
        let cancelled =
            tokio::time::timeout(Duration::from_millis(50), port.read_line(&mut line)).await;
        assert!(cancelled.is_err(), "no newline yet: the read must wait");

        host.write_all(b"\"x\"}\n").unwrap();
        assert_eq!(line_of(&mut port).await, "{\"op\":\"x\"}\n");
    }

    #[tokio::test]
    async fn port_splits_lines_across_reads() {
        let fifo = Fifo::new("split");
        let mut port = fifo.reader();
        let mut host = fifo.host();

        host.write_all(b"one\ntwo\nthr").unwrap();
        assert_eq!(line_of(&mut port).await, "one\n");
        assert_eq!(line_of(&mut port).await, "two\n");
        host.write_all(b"ee\n").unwrap();
        assert_eq!(line_of(&mut port).await, "three\n");
    }

    // -------------------------------------------------------------------------
    // answer_ping
    // -------------------------------------------------------------------------

    /// The messages that answer_ping queues for `host_entropy`, and the host entropy its kernel
    /// step got (the step returns `step_result`; entropy.rs tests the real step).
    async fn ping_answers(
        host_entropy: Option<&str>,
        step_result: Result<(), String>,
    ) -> (Vec<serde_json::Value>, Option<String>) {
        let (tx, mut rx) = mpsc::channel(4);
        let writer = ResponseWriter::new(tx, Some("op-1".to_string()));
        let mut stepped = None;
        answer_ping(&writer, host_entropy, |entropy| {
            stepped = Some(entropy.to_string());
            step_result
        })
        .await
        .unwrap();
        let mut answers = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            answers.push(serde_json::from_slice(&frame).unwrap());
        }
        (answers, stepped)
    }

    #[tokio::test]
    async fn a_ping_without_host_entropy_gets_a_pong_and_no_kernel_step() {
        let (answers, stepped) = ping_answers(None, Ok(())).await;

        assert_eq!(
            answers,
            vec![serde_json::json!({"type": "pong", "version": VERSION, "op_id": "op-1"})]
        );
        assert_eq!(stepped, None);
    }

    #[tokio::test]
    async fn a_ping_with_host_entropy_runs_the_kernel_step_before_its_pong() {
        let (answers, stepped) = ping_answers(Some("AAEC"), Ok(())).await;

        assert_eq!(
            answers,
            vec![serde_json::json!({"type": "pong", "version": VERSION, "op_id": "op-1"})]
        );
        assert_eq!(stepped.as_deref(), Some("AAEC"));
    }

    #[tokio::test]
    async fn a_failed_kernel_step_answers_the_ping_with_an_error_and_no_pong() {
        let failure = Err("ioctl RNDRESEEDCRNG failed: no data".to_string());

        let (answers, _) = ping_answers(Some("AAEC"), failure).await;

        assert_eq!(
            answers,
            vec![serde_json::json!({
                "type": "error",
                "message": "kernel random reseed with the host entropy failed: ioctl RNDRESEEDCRNG failed: no data",
                "error_type": "io_error",
                "op_id": "op-1",
            })]
        );
    }

    #[tokio::test]
    async fn a_closed_queue_fails_both_answers_of_a_ping() {
        for host_entropy in [None, Some("AAEC")] {
            let (tx, rx) = mpsc::channel(1);
            drop(rx);

            let sent = answer_ping(&ResponseWriter::new(tx, None), host_entropy, |_| {
                Err("no data".to_string())
            })
            .await;

            assert!(sent.is_err(), "host_entropy={host_entropy:?}");
        }
    }

    // -------------------------------------------------------------------------
    // finish_write
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn finish_write_abandons_a_task_that_does_not_finish() {
        let (chunk_tx, _chunk_rx) = mpsc::channel::<Vec<u8>>(16);
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        // A writer stuck on the disk: it returns only when the test lets it.
        let task = tokio::task::spawn_blocking(move || {
            let _ = release_rx.recv();
            Err("released".to_string())
        });
        let end = WriteEnd::new();

        let finished = tokio::time::timeout(
            Duration::from_secs(2),
            finish_write(chunk_tx, task, &end, Duration::from_millis(50)),
        )
        .await
        .expect("finish_write must give up at its limit");

        assert!(matches!(finished, WriteFinish::Abandoned));
        assert!(!end.try_commit(), "an abandoned write cannot commit");
        release_tx.send(()).unwrap();
    }

    #[tokio::test]
    async fn finish_write_waits_for_a_task_that_claimed_the_commit() {
        // The task claimed the commit, then its rename is slow: past the limit the
        // agent must report the write's own result, not abandon one that lands.
        // The claim comes first, so no timing decides the test.
        let (chunk_tx, _chunk_rx) = mpsc::channel::<Vec<u8>>(16);
        let end = WriteEnd::new();
        assert!(end.try_commit());
        // Done in the second wait: 750 ms, between the limit (500 ms) and twice it
        let task = tokio::task::spawn_blocking(move || {
            std::thread::sleep(Duration::from_millis(750)); // the rename, slow
            Ok(2)
        });

        let finished = finish_write(chunk_tx, task, &end, Duration::from_millis(500)).await;

        let WriteFinish::Done(Ok(Ok(bytes_written))) = finished else {
            panic!("a committed write is not abandoned");
        };
        assert_eq!(bytes_written, 2);
    }

    #[tokio::test]
    async fn finish_write_gives_up_on_a_commit_that_stalls() {
        // The task claimed the commit, then its rename never ends (a stalled disk):
        // past both waits the agent goes on and says the write may still land.
        let (chunk_tx, _chunk_rx) = mpsc::channel::<Vec<u8>>(16);
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let end = WriteEnd::new();
        assert!(end.try_commit());
        let task = tokio::task::spawn_blocking(move || {
            let _ = release_rx.recv();
            Ok(2)
        });

        let finished = tokio::time::timeout(
            Duration::from_secs(2),
            finish_write(chunk_tx, task, &end, Duration::from_millis(50)),
        )
        .await
        .expect("finish_write must give up at the end of its second wait");

        assert!(matches!(finished, WriteFinish::CommitStalled));
        release_tx.send(()).unwrap();
    }

    #[tokio::test]
    async fn finish_write_returns_the_result_of_a_task_that_finishes() {
        let (chunk_tx, mut chunk_rx) = mpsc::channel::<Vec<u8>>(16);
        let task = tokio::task::spawn_blocking(move || {
            assert_eq!(chunk_rx.blocking_recv(), Some(vec![]), "end marker");
            Ok(4)
        });
        let end = WriteEnd::new();

        let finished = finish_write(chunk_tx, task, &end, Duration::from_secs(2)).await;

        let WriteFinish::Done(Ok(Ok(bytes_written))) = finished else {
            panic!("the task finished");
        };
        assert_eq!(bytes_written, 4);
        assert!(end.try_commit(), "the write was not abandoned");
    }

    // -------------------------------------------------------------------------
    // extract_op_id
    // -------------------------------------------------------------------------

    #[test]
    fn op_id_present() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"action":"ping","op_id":"abc"}"#).unwrap();
        assert_eq!(extract_op_id(&v), Some("abc".into()));
    }

    #[test]
    fn op_id_missing() {
        let v: serde_json::Value = serde_json::from_str(r#"{"action":"ping"}"#).unwrap();
        assert_eq!(extract_op_id(&v), None);
    }

    #[test]
    fn op_id_null() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"action":"ping","op_id":null}"#).unwrap();
        assert_eq!(extract_op_id(&v), None);
    }

    #[test]
    fn op_id_empty_string() {
        let v: serde_json::Value = serde_json::from_str(r#"{"action":"ping","op_id":""}"#).unwrap();
        assert_eq!(extract_op_id(&v), Some("".into()));
    }

    #[test]
    fn op_id_non_string_int() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"action":"ping","op_id":123}"#).unwrap();
        assert_eq!(extract_op_id(&v), None);
    }

    #[test]
    fn op_id_from_file_command() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"action":"write_file","op_id":"abc","path":"x","make_executable":false}"#,
        )
        .unwrap();
        assert_eq!(extract_op_id(&v), Some("abc".into()));
        // Also verify the command still deserializes correctly
        let cmd: GuestCommand = serde_json::from_value(v).unwrap();
        match cmd {
            GuestCommand::WriteFile { op_id, path, .. } => {
                assert_eq!(op_id, "abc");
                assert_eq!(path, "x");
            }
            _ => panic!("expected WriteFile"),
        }
    }
}

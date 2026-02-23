//! Connection handling: non-blocking virtio-serial I/O and command dispatch.

use std::collections::HashMap;
use std::os::unix::io::{AsRawFd, RawFd};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tokio::io::unix::AsyncFd;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::mpsc;

use crate::constants::*;
use crate::error::*;
use crate::file_io::{ActiveWriteHandle, handle_list_files, handle_read_file, handle_write_file};
use crate::packages::install_packages;
use crate::repl::execute::execute_code_streaming;
use crate::repl::spawn::spawn_repl;
use crate::types::{GuestCommand, GuestResponse, Language};

// ============================================================================
// NonBlockingFile
// ============================================================================

/// Non-blocking file wrapper for virtio-serial ports.
///
/// Uses AsyncFd for true async I/O (epoll-based) instead of tokio::fs::File
/// which uses a blocking threadpool.
pub(crate) struct NonBlockingFile {
    async_fd: AsyncFd<std::fs::File>,
    leftover: Vec<u8>,
}

impl NonBlockingFile {
    pub(crate) fn open_read(path: &str) -> std::io::Result<Self> {
        use std::fs::OpenOptions;
        let file = OpenOptions::new().read(true).open(path)?;
        Self::set_nonblocking(file.as_raw_fd())?;
        let async_fd = AsyncFd::new(file)?;
        Ok(Self {
            async_fd,
            leftover: Vec::new(),
        })
    }

    fn set_nonblocking(fd: RawFd) -> std::io::Result<()> {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
        if result < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    /// Check if the host side of the virtio-serial port is connected.
    pub(crate) fn is_host_connected(&self) -> std::io::Result<bool> {
        let fd = self.async_fd.get_ref().as_raw_fd();
        let mut pollfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let result = unsafe { libc::poll(&mut pollfd, 1, 0) };
        if result < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok((pollfd.revents & libc::POLLHUP) == 0)
    }

    /// Read a line with proper async timeout support and buffered I/O.
    pub(crate) async fn read_line(&mut self, buf: &mut String) -> std::io::Result<usize> {
        let mut total_bytes = 0;
        let mut bytes = Vec::new();

        if !self.leftover.is_empty() {
            if let Some(pos) = self.leftover.iter().position(|&b| b == b'\n') {
                let line_len = pos + 1;
                buf.push_str(&String::from_utf8_lossy(&self.leftover[..line_len]));
                self.leftover.drain(..line_len);
                return Ok(line_len);
            }
            total_bytes += self.leftover.len();
            bytes.append(&mut self.leftover);
        }

        let mut read_buf = [0u8; 16384];

        loop {
            let mut guard = self.async_fd.readable().await?;

            match guard.try_io(|inner| {
                let fd = inner.get_ref().as_raw_fd();
                let result = unsafe {
                    libc::read(
                        fd,
                        read_buf.as_mut_ptr() as *mut libc::c_void,
                        read_buf.len(),
                    )
                };
                if result < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(result as usize)
                }
            }) {
                Ok(Ok(0)) => {
                    buf.push_str(&String::from_utf8_lossy(&bytes));
                    return Ok(total_bytes);
                }
                Ok(Ok(n)) => {
                    let chunk = &read_buf[..n];
                    if let Some(pos) = chunk.iter().position(|&b| b == b'\n') {
                        bytes.extend_from_slice(&chunk[..=pos]);
                        total_bytes += pos + 1;
                        if pos + 1 < n {
                            self.leftover.extend_from_slice(&chunk[pos + 1..]);
                        }
                        buf.push_str(&String::from_utf8_lossy(&bytes));
                        return Ok(total_bytes);
                    }
                    total_bytes += n;
                    bytes.extend_from_slice(chunk);
                }
                Ok(Err(e)) => return Err(e),
                Err(_would_block) => continue,
            }
        }
    }
}

// ============================================================================
// Connection handler
// ============================================================================

/// Handle connection with non-blocking command reader.
pub(crate) async fn handle_connection(
    cmd_reader: &mut NonBlockingFile,
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
                eprintln!("Write error: {e}");
                break;
            }
            pending += 1;
            if pending >= 16 || write_rx.is_empty() {
                if let Err(e) = writer.flush().await {
                    eprintln!("Flush error: {e}");
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

        let read_result = tokio::time::timeout(
            std::time::Duration::from_millis(READ_TIMEOUT_MS),
            cmd_reader.read_line(&mut line),
        )
        .await;

        let bytes_read = match read_result {
            Ok(Ok(0)) => {
                eprintln!("Connection closed by client");
                break Ok(());
            }
            Ok(Ok(n)) => n,
            Ok(Err(e)) => {
                eprintln!("Read error: {e}");
                break Err(e.into());
            }
            Err(_) => {
                eprintln!("Read timeout after {READ_TIMEOUT_MS}ms, reconnecting...");
                break Err("read timeout - triggering reconnect".into());
            }
        };

        if bytes_read > MAX_REQUEST_SIZE_BYTES {
            let _ = send_streaming_error(
                &write_tx,
                format!(
                    "Request too large: {} bytes (max {} bytes)",
                    bytes_read, MAX_REQUEST_SIZE_BYTES
                ),
                "request_error",
                None,
            )
            .await;
            continue;
        }

        eprintln!("Received request ({bytes_read} bytes)");

        match serde_json::from_str::<GuestCommand>(&line) {
            Ok(GuestCommand::Ping) => {
                eprintln!("Processing: ping");
                if send_response(&write_tx, &GuestResponse::Pong { version: VERSION })
                    .await
                    .is_err()
                {
                    break Err("write queue closed".into());
                }
            }
            Ok(GuestCommand::WarmRepl { language }) => {
                eprintln!("Processing: warm_repl (language={language})");
                let lang = match Language::parse(&language) {
                    Some(l) => l,
                    None => {
                        if send_response(
                            &write_tx,
                            &GuestResponse::WarmReplAck {
                                language,
                                status: "error".to_string(),
                                message: Some(
                                    "Unsupported language (supported: python, javascript, raw)"
                                        .to_string(),
                                ),
                            },
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                        continue;
                    }
                };
                match spawn_repl(lang).await {
                    Ok(repl) => {
                        REPL_STATES.lock().await.insert(lang, repl);
                        eprintln!("REPL pre-warmed for {language}");
                        if send_response(
                            &write_tx,
                            &GuestResponse::WarmReplAck {
                                language,
                                status: "ok".to_string(),
                                message: None,
                            },
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: eager REPL spawn failed: {e}");
                        if send_response(
                            &write_tx,
                            &GuestResponse::WarmReplAck {
                                language,
                                status: "error".to_string(),
                                message: Some(e.to_string()),
                            },
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                }
            }
            Ok(GuestCommand::InstallPackages { language, packages }) => {
                eprintln!(
                    "Processing: install_packages (language={language}, count={})",
                    packages.len()
                );
                let lang = match Language::parse(&language) {
                    Some(l) => l,
                    None => {
                        if send_streaming_error(
                            &write_tx,
                            format!(
                                "Unsupported language '{}' for package installation (supported: python, javascript)",
                                language
                            ),
                            "validation_error",
                            None,
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                        continue;
                    }
                };
                match install_packages(lang, &packages, &write_tx).await {
                    Ok(()) => {}
                    Err(CmdError::Reply {
                        message,
                        error_type,
                        op_id,
                    }) => {
                        if send_streaming_error(&write_tx, message, error_type, op_id.as_deref())
                            .await
                            .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                    Err(CmdError::Fatal(e)) => {
                        break Err(e.to_string().into());
                    }
                }
            }
            Ok(GuestCommand::ExecuteCode {
                language,
                code,
                timeout,
                env_vars,
            }) => {
                eprintln!(
                    "Processing: execute_code (language={language}, code_size={}, timeout={timeout}s, env_vars={})",
                    code.len(),
                    env_vars.len()
                );
                match execute_code_streaming(&language, &code, timeout, &env_vars, &write_tx).await
                {
                    Ok(()) => {}
                    Err(CmdError::Reply {
                        message,
                        error_type,
                        op_id,
                    }) => {
                        if send_streaming_error(&write_tx, message, error_type, op_id.as_deref())
                            .await
                            .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                    Err(CmdError::Fatal(e)) => {
                        break Err(e.to_string().into());
                    }
                }
            }
            Ok(GuestCommand::WriteFile {
                op_id,
                path,
                make_executable,
            }) => {
                eprintln!(
                    "Processing: write_file (op_id={op_id}, path={path}, executable={make_executable})"
                );
                #[allow(clippy::map_entry)] // Entry API incompatible with async error path
                if active_writes.contains_key(&op_id) {
                    let _ = send_streaming_error(
                        &write_tx,
                        format!("Duplicate op_id '{op_id}' for write_file"),
                        "validation_error",
                        Some(&op_id),
                    )
                    .await;
                } else {
                    match handle_write_file(&op_id, &path, make_executable, &write_tx).await {
                        Ok(handle) => {
                            active_writes.insert(op_id, handle);
                        }
                        Err(CmdError::Reply {
                            message,
                            error_type,
                            op_id: err_op_id,
                        }) => {
                            let _ = send_streaming_error(
                                &write_tx,
                                message,
                                error_type,
                                err_op_id.as_deref(),
                            )
                            .await;
                        }
                        Err(CmdError::Fatal(e)) => {
                            break Err(e.to_string().into());
                        }
                    }
                }
            }
            Ok(GuestCommand::FileChunk { op_id, data }) => {
                if !active_writes.contains_key(&op_id) {
                    let _ = send_streaming_error(
                        &write_tx,
                        format!("No active write for op_id '{op_id}'"),
                        "protocol_error",
                        Some(&op_id),
                    )
                    .await;
                    continue;
                }
                let decoded = match BASE64.decode(&data) {
                    Ok(d) => d,
                    Err(e) => {
                        let handle = active_writes.remove(&op_id).unwrap();
                        drop(handle.chunk_tx);
                        let _ = send_streaming_error(
                            &write_tx,
                            format!(
                                "Invalid base64 in chunk for '{}': {}",
                                handle.path_display, e
                            ),
                            "validation_error",
                            Some(&op_id),
                        )
                        .await;
                        continue;
                    }
                };
                let send_failed = {
                    let handle = active_writes.get(&op_id).unwrap();
                    handle.chunk_tx.send(decoded).await.is_err()
                };
                if send_failed {
                    let handle = active_writes.remove(&op_id).unwrap();
                    drop(handle.chunk_tx);
                    match handle.task.await {
                        Ok(Err(write_err)) => {
                            eprintln!(
                                "Write pipeline error for '{}' (op_id={}): {}",
                                write_err.path_display, write_err.op_id, write_err.message
                            );
                            let _ = send_streaming_error(
                                &write_tx,
                                write_err.message,
                                &write_err.error_type,
                                Some(&write_err.op_id),
                            )
                            .await;
                        }
                        Ok(Ok(_)) => {
                            let _ = send_streaming_error(
                                &write_tx,
                                format!(
                                    "Write pipeline unexpectedly closed for '{}'",
                                    handle.path_display
                                ),
                                "io_error",
                                Some(&op_id),
                            )
                            .await;
                        }
                        Err(join_err) => {
                            let _ = send_streaming_error(
                                &write_tx,
                                format!(
                                    "Internal error writing '{}': {}",
                                    handle.path_display, join_err
                                ),
                                "io_error",
                                Some(&handle.op_id),
                            )
                            .await;
                        }
                    }
                    continue;
                }
            }
            Ok(GuestCommand::FileEnd { op_id }) => {
                let ActiveWriteHandle {
                    chunk_tx,
                    task,
                    path_display,
                    op_id: handle_op_id,
                } = match active_writes.remove(&op_id) {
                    Some(h) => h,
                    None => {
                        let _ = send_streaming_error(
                            &write_tx,
                            format!("No active write for op_id '{op_id}'"),
                            "protocol_error",
                            Some(&op_id),
                        )
                        .await;
                        continue;
                    }
                };
                let _ = chunk_tx.send(vec![]).await;
                drop(chunk_tx);
                match task.await {
                    Ok(Ok(result)) => {
                        if send_response(
                            &write_tx,
                            &GuestResponse::FileWriteAck {
                                op_id: result.op_id,
                                path: result.path_display,
                                bytes_written: result.bytes_written,
                            },
                        )
                        .await
                        .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                    Ok(Err(write_err)) => {
                        eprintln!(
                            "Write pipeline error for '{}' (op_id={}): {}",
                            write_err.path_display, write_err.op_id, write_err.message
                        );
                        let _ = send_streaming_error(
                            &write_tx,
                            write_err.message,
                            &write_err.error_type,
                            Some(&write_err.op_id),
                        )
                        .await;
                    }
                    Err(join_err) => {
                        let _ = send_streaming_error(
                            &write_tx,
                            format!("Internal error writing '{}': {}", path_display, join_err),
                            "io_error",
                            Some(&handle_op_id),
                        )
                        .await;
                    }
                }
            }
            Ok(GuestCommand::ReadFile { op_id, path }) => {
                eprintln!("Processing: read_file (op_id={op_id}, path={path})");
                match handle_read_file(&op_id, &path, &write_tx).await {
                    Ok(()) => {}
                    Err(CmdError::Reply {
                        message,
                        error_type,
                        op_id,
                    }) => {
                        if send_streaming_error(&write_tx, message, error_type, op_id.as_deref())
                            .await
                            .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                    Err(CmdError::Fatal(e)) => {
                        break Err(e.to_string().into());
                    }
                }
            }
            Ok(GuestCommand::ListFiles { path }) => {
                eprintln!("Processing: list_files (path={path})");
                match handle_list_files(&path, &write_tx).await {
                    Ok(()) => {}
                    Err(CmdError::Reply {
                        message,
                        error_type,
                        op_id,
                    }) => {
                        if send_streaming_error(&write_tx, message, error_type, op_id.as_deref())
                            .await
                            .is_err()
                        {
                            break Err("write queue closed".into());
                        }
                    }
                    Err(CmdError::Fatal(e)) => {
                        break Err(e.to_string().into());
                    }
                }
            }
            Err(e) => {
                eprintln!("JSON parse error: {e}");
                let _ = send_streaming_error(
                    &write_tx,
                    format!("Invalid JSON: {e}"),
                    "request_error",
                    None,
                )
                .await;
            }
        }
    };

    drop(write_tx);
    let _ = write_handle.await;
    result
}

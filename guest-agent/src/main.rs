//! QEMU Guest Agent
//!
//! Lightweight async agent running inside QEMU microVMs.
//! Communicates with host via virtio-serial for:
//! - Package installation (pip, npm)
//! - Code execution
//! - Health checks
//!
//! Uses tokio for fully async, non-blocking I/O.
//! Communication via dual virtio-serial ports:
//! - /dev/virtio-ports/org.dualeai.cmd (host → guest, read-only)
//! - /dev/virtio-ports/org.dualeai.event (guest → host, write-only)

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command as StdCommand;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::Command;
use tokio::sync::mpsc;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const CMD_PORT_PATH: &str = "/dev/virtio-ports/org.dualeai.cmd";
const EVENT_PORT_PATH: &str = "/dev/virtio-ports/org.dualeai.event";

// Execution limits
const MAX_CODE_SIZE_BYTES: usize = 1_000_000; // 1MB max code size
const MAX_PACKAGE_OUTPUT_BYTES: usize = 50_000; // 50KB max package install output
const MAX_TIMEOUT_SECONDS: u64 = 300; // 5 minutes max execution timeout

// Connection limits
const MAX_REQUEST_SIZE_BYTES: usize = 2_000_000; // 2MB max request JSON
const RETRY_DELAY_MS: u64 = 50; // 50ms retry delay on transient errors
const WRITE_QUEUE_SIZE: usize = 100; // Bounded channel size for write queue (prevents deadlocks)

// Environment variable limits
const MAX_ENV_VARS: usize = 100; // Max number of environment variables
const MAX_ENV_VAR_NAME_LENGTH: usize = 256; // Max env var name length
const MAX_ENV_VAR_VALUE_LENGTH: usize = 4096; // Max env var value length

// Package limits
const MAX_PACKAGES: usize = 50; // Max number of packages per install
const MAX_PACKAGE_NAME_LENGTH: usize = 214; // Max package name length (PyPI limit)
const PACKAGE_INSTALL_TIMEOUT_SECONDS: u64 = 300; // 5 min timeout for package installs

// Streaming configuration (Jan 2026 best practice)
// - 50ms flush interval for low-latency real-time feel
// - 64KB max buffer to prevent memory exhaustion
// - Backpressure via bounded channel when buffer full
const FLUSH_INTERVAL_MS: u64 = 50; // 50ms flush interval (not 1s - too slow for real-time)
const MAX_BUFFER_SIZE_BYTES: usize = 64 * 1024; // 64KB max buffer before forced flush

/// Regex for validating package names with version specifiers (required).
///
/// Pattern: Package name + version operator (required) + version spec
/// - Package name: [a-zA-Z0-9_\-\.]+
/// - Version operator: [@=<>~] (at least one required)
/// - Version spec: [a-zA-Z0-9_\-\.@/=<>~\^\*\[\], ]*
///
/// Supports:
/// - npm: lodash@4.17.21, lodash@~4.17, lodash@^4.0.0
/// - Python: pandas==2.0.0, pandas~=2.0, pandas>=2.0,<3.0
///
/// Rejects packages without version: "pandas", "lodash"
static PACKAGE_NAME_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[a-zA-Z0-9_\-\.]+[@=<>~][a-zA-Z0-9_\-\.@/=<>~\^\*\[\], ]*$").unwrap()
});

/// Blacklist of dangerous environment variables.
///
/// Security rationale:
/// - LD_PRELOAD/LD_LIBRARY_PATH/LD_AUDIT: Arbitrary code execution via library injection
/// - BASH_ENV/ENV: Execute arbitrary file on shell startup
/// - PATH: Executable search path manipulation (could bypass sandboxing)
/// - GCONV_PATH: glibc converter modules (code injection)
/// - HOSTALIASES: DNS resolution manipulation
/// - PROMPT_COMMAND: Execute arbitrary commands in bash
/// - MALLOC_*: Memory allocator hooks (potential exploitation)
/// - NODE_OPTIONS: Node.js runtime options (can execute arbitrary code)
/// - PYTHONWARNINGS/PYTHONSTARTUP: Python module injection
/// - GLIBC_TUNABLES: CVE-2023-4911 buffer overflow
static BLOCKED_ENV_VARS: &[&str] = &[
    // Dynamic linker (arbitrary code execution)
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "LD_BIND_NOW",
    "LD_DEBUG",
    "LD_DEBUG_OUTPUT",
    "LD_USE_LOAD_BIAS",
    "LD_PROFILE",
    "LD_ORIGIN_PATH",
    "LD_AOUT_LIBRARY_PATH",
    "LD_AOUT_PRELOAD",
    // glibc tunables - CVE-2023-4911
    "GLIBC_TUNABLES",
    // Node.js runtime
    "NODE_OPTIONS",
    "NODE_REPL_HISTORY",
    // Python runtime
    "PYTHONWARNINGS",
    "PYTHONSTARTUP",
    "PYTHONHOME",
    // Shell environment execution
    "BASH_ENV",
    "ENV",
    "PROMPT_COMMAND",
    // Path manipulation
    "PATH",
    // glibc/system hooks
    "GCONV_PATH",
    "HOSTALIASES",
    "LOCPATH",
    "NLSPATH",
    "RESOLV_HOST_CONF",
    "RES_OPTIONS",
    "TMPDIR",
    "TZDIR",
    "MALLOC_CHECK_",
    "MALLOC_TRACE",
    "MALLOC_PERTURB_",
];

#[derive(Debug, Deserialize)]
#[serde(tag = "action")]
enum GuestCommand {
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
}

#[derive(Debug, Serialize)]
struct OutputChunk {
    #[serde(rename = "type")]
    chunk_type: String, // "stdout" or "stderr"
    chunk: String,
}

#[derive(Debug, Serialize)]
struct ExecutionComplete {
    #[serde(rename = "type")]
    msg_type: String, // "complete"
    exit_code: i32,
    execution_time_ms: u64,
}

#[derive(Debug, Serialize)]
struct Pong {
    #[serde(rename = "type")]
    msg_type: String, // "pong"
    version: String,
}

#[derive(Debug, Serialize)]
struct StreamingError {
    #[serde(rename = "type")]
    msg_type: String, // "error"
    message: String,
    error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
}

// Helper to send streaming error via queue
async fn send_streaming_error(
    write_tx: &mpsc::Sender<Vec<u8>>,
    message: String,
    error_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let error = StreamingError {
        msg_type: "error".to_string(),
        message,
        error_type: error_type.to_string(),
        version: Some(VERSION.to_string()),
    };
    let json = serde_json::to_string(&error)?;
    let mut response = json.into_bytes();
    response.push(b'\n');

    // Queue write (blocks if queue full - backpressure)
    write_tx
        .send(response)
        .await
        .map_err(|_| "Write queue closed")?;
    Ok(())
}

// Note: flush_buffers() removed - replaced with spawned task pattern (Nov 2025)

async fn install_packages(
    language: &str,
    packages: &[String],
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    use tokio::time::Duration;

    // Validation: check language is supported
    if language != "python" && language != "javascript" {
        send_streaming_error(
            write_tx,
            format!(
                "Unsupported language '{}' for package installation (supported: python, javascript)",
                language
            ),
            "validation_error",
        ).await?;
        return Ok(());
    }

    // Validation: check packages list is not empty
    if packages.is_empty() {
        send_streaming_error(
            write_tx,
            "No packages specified for installation".to_string(),
            "validation_error",
        )
        .await?;
        return Ok(());
    }

    // Validation: check package count
    if packages.len() > MAX_PACKAGES {
        send_streaming_error(
            write_tx,
            format!(
                "Too many packages: {} (max {})",
                packages.len(),
                MAX_PACKAGES
            ),
            "validation_error",
        )
        .await?;
        return Ok(());
    }

    // Validation: check for suspicious package names
    for pkg in packages {
        // Check empty
        if pkg.is_empty() {
            send_streaming_error(
                write_tx,
                "Package name cannot be empty".to_string(),
                "validation_error",
            )
            .await?;
            return Ok(());
        }

        // Check length
        if pkg.len() > MAX_PACKAGE_NAME_LENGTH {
            send_streaming_error(
                write_tx,
                format!(
                    "Package name too long: {} bytes (max {})",
                    pkg.len(),
                    MAX_PACKAGE_NAME_LENGTH
                ),
                "validation_error",
            )
            .await?;
            return Ok(());
        }

        // Check for path traversal and suspicious characters
        if pkg.contains("..") || pkg.contains("/") || pkg.contains("\\") {
            send_streaming_error(
                write_tx,
                format!(
                    "Invalid package name: '{}' (path characters not allowed)",
                    pkg
                ),
                "validation_error",
            )
            .await?;
            return Ok(());
        }

        // Check for null bytes
        if pkg.contains('\0') {
            send_streaming_error(
                write_tx,
                "Package name contains null byte".to_string(),
                "validation_error",
            )
            .await?;
            return Ok(());
        }

        // Check for control characters
        if pkg.chars().any(|c| c.is_control()) {
            send_streaming_error(
                write_tx,
                format!(
                    "Invalid package name: '{}' (control characters not allowed)",
                    pkg
                ),
                "validation_error",
            )
            .await?;
            return Ok(());
        }

        // Check against regex (allows alphanumeric, dash, underscore, dot, @, /, =, <, >, ~, [, ])
        if !PACKAGE_NAME_REGEX.is_match(pkg) {
            send_streaming_error(
                write_tx,
                format!(
                    "Invalid package name: '{}' (contains invalid characters)",
                    pkg
                ),
                "validation_error",
            )
            .await?;
            return Ok(());
        }
    }

    let start = Instant::now();

    let mut cmd = match language {
        "python" => {
            let mut c = Command::new("uv");
            c.arg("pip")
                .arg("install")
                .arg("--system")
                .arg("--break-system-packages");
            for pkg in packages {
                c.arg(pkg);
            }
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            c
        }
        "javascript" => {
            let mut c = Command::new("bun");
            c.arg("add").arg("--global");
            for pkg in packages {
                c.arg(pkg);
            }
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            c
        }
        _ => unreachable!(), // Already validated above
    };

    // Spawn process
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Failed to execute package manager for {}: {}", language, e),
                "execution_error",
            )
            .await?;
            return Ok(());
        }
    };

    // Get stdout/stderr streams
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Spawn independent reader tasks (Tokio best practice Nov 2025 - ZERO race conditions)
    // These tasks continue reading until EOF regardless of when child.wait() completes
    let stdout_task = tokio::spawn(async move {
        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut lines = Vec::new();
        let mut total_bytes = 0usize;

        while let Ok(Some(line)) = stdout_reader.next_line().await {
            if total_bytes + line.len() + 1 > MAX_PACKAGE_OUTPUT_BYTES {
                let remaining = MAX_PACKAGE_OUTPUT_BYTES.saturating_sub(total_bytes);
                if remaining > 0 {
                    lines.push(line[..remaining.min(line.len())].to_string());
                }
                lines.push(format!(
                    "[truncated: output limit {}KB exceeded]",
                    MAX_PACKAGE_OUTPUT_BYTES / 1024
                ));
                break;
            }
            total_bytes += line.len() + 1;
            lines.push(line);
        }
        lines
    });

    let stderr_task = tokio::spawn(async move {
        let mut stderr_reader = BufReader::new(stderr).lines();
        let mut lines = Vec::new();
        let mut total_bytes = 0usize;

        while let Ok(Some(line)) = stderr_reader.next_line().await {
            if total_bytes + line.len() + 1 > MAX_PACKAGE_OUTPUT_BYTES {
                let remaining = MAX_PACKAGE_OUTPUT_BYTES.saturating_sub(total_bytes);
                if remaining > 0 {
                    lines.push(line[..remaining.min(line.len())].to_string());
                }
                lines.push(format!(
                    "[truncated: output limit {}KB exceeded]",
                    MAX_PACKAGE_OUTPUT_BYTES / 1024
                ));
                break;
            }
            total_bytes += line.len() + 1;
            lines.push(line);
        }
        lines
    });

    // Wait for process with timeout
    let wait_result = tokio::time::timeout(
        Duration::from_secs(PACKAGE_INSTALL_TIMEOUT_SECONDS),
        child.wait(),
    )
    .await;

    let status = match wait_result {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            let _ = child.kill().await;
            send_streaming_error(
                write_tx,
                format!("Process wait error: {}", e),
                "execution_error",
            )
            .await?;
            return Ok(());
        }
        Err(_) => {
            let _ = child.kill().await;
            send_streaming_error(
                write_tx,
                format!(
                    "Package installation timeout after {}s",
                    PACKAGE_INSTALL_TIMEOUT_SECONDS
                ),
                "timeout_error",
            )
            .await?;
            return Ok(());
        }
    };

    // Reader tasks continue independently - guaranteed to capture ALL output
    let stdout_lines = stdout_task.await.unwrap_or_default();
    let stderr_lines = stderr_task.await.unwrap_or_default();

    let duration_ms = start.elapsed().as_millis() as u64;
    let exit_code = status.code().unwrap_or(-1);

    // Stream all captured output (batched for efficiency)
    if !stdout_lines.is_empty() {
        let chunk = OutputChunk {
            chunk_type: "stdout".to_string(),
            chunk: stdout_lines.join("\n") + "\n",
        };
        let json = serde_json::to_string(&chunk)?;
        let mut response = json.into_bytes();
        response.push(b'\n');
        write_tx
            .send(response)
            .await
            .map_err(|_| "Write queue closed")?;
    }

    if !stderr_lines.is_empty() {
        let chunk = OutputChunk {
            chunk_type: "stderr".to_string(),
            chunk: stderr_lines.join("\n") + "\n",
        };
        let json = serde_json::to_string(&chunk)?;
        let mut response = json.into_bytes();
        response.push(b'\n');
        write_tx
            .send(response)
            .await
            .map_err(|_| "Write queue closed")?;
    }

    // Send completion message
    let complete = ExecutionComplete {
        msg_type: "complete".to_string(),
        exit_code,
        execution_time_ms: duration_ms,
    };
    let json = serde_json::to_string(&complete)?;
    let mut response = json.into_bytes();
    response.push(b'\n');
    write_tx
        .send(response)
        .await
        .map_err(|_| "Write queue closed")?;

    Ok(())
}

// Validation helper for execute_code_streaming
fn validate_execute_params(
    language: &str,
    code: &str,
    timeout: u64,
    env_vars: &HashMap<String, String>,
) -> Result<(), String> {
    // Check language
    if language != "python" && language != "javascript" && language != "raw" {
        return Err(format!(
            "Unsupported language '{}' (supported: python, javascript, raw)",
            language
        ));
    }

    // Check code not empty
    if code.trim().is_empty() {
        return Err("Code cannot be empty".to_string());
    }

    // Check code size
    if code.len() > MAX_CODE_SIZE_BYTES {
        return Err(format!(
            "Code too large: {} bytes (max {} bytes)",
            code.len(),
            MAX_CODE_SIZE_BYTES
        ));
    }

    // Check timeout
    if timeout > MAX_TIMEOUT_SECONDS {
        return Err(format!(
            "Timeout too large: {}s (max {}s)",
            timeout, MAX_TIMEOUT_SECONDS
        ));
    }

    // Check environment variables count
    if env_vars.len() > MAX_ENV_VARS {
        return Err(format!(
            "Too many environment variables: {} (max {})",
            env_vars.len(),
            MAX_ENV_VARS
        ));
    }

    // Check each env var
    for (key, value) in env_vars {
        if BLOCKED_ENV_VARS.contains(&key.to_uppercase().as_str()) {
            return Err(format!(
                "Blocked environment variable: '{}' (security risk)",
                key
            ));
        }

        if key.is_empty() || key.len() > MAX_ENV_VAR_NAME_LENGTH {
            return Err(format!(
                "Invalid environment variable name length: {} (max {})",
                key.len(),
                MAX_ENV_VAR_NAME_LENGTH
            ));
        }

        if value.len() > MAX_ENV_VAR_VALUE_LENGTH {
            return Err(format!(
                "Environment variable value too large: {} bytes (max {})",
                value.len(),
                MAX_ENV_VAR_VALUE_LENGTH
            ));
        }

        if key.contains('\0') || value.contains('\0') {
            return Err("Environment variable contains null byte".to_string());
        }
    }

    Ok(())
}

/// Helper to flush a buffer as an OutputChunk message
async fn flush_output_buffer(
    write_tx: &mpsc::Sender<Vec<u8>>,
    buffer: &mut String,
    chunk_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if buffer.is_empty() {
        return Ok(());
    }

    let chunk = OutputChunk {
        chunk_type: chunk_type.to_string(),
        chunk: std::mem::take(buffer),
    };
    let json = serde_json::to_string(&chunk)?;
    let mut response = json.into_bytes();
    response.push(b'\n');
    write_tx
        .send(response)
        .await
        .map_err(|_| "Write queue closed")?;
    Ok(())
}

async fn execute_code_streaming(
    language: &str,
    code: &str,
    timeout: u64,
    env_vars: &HashMap<String, String>,
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    use tokio::io::AsyncReadExt;
    use tokio::time::{interval, Duration};

    // Validate params
    if let Err(error_message) = validate_execute_params(language, code, timeout, env_vars) {
        send_streaming_error(write_tx, error_message, "validation_error").await?;
        return Ok(());
    }

    let start = Instant::now();

    let mut cmd = match language {
        "python" => {
            let mut c = Command::new("python3");
            c.arg("-c").arg(code);
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            c
        }
        "javascript" => {
            let mut c = Command::new("bun");
            c.arg("-e").arg(code);
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            c
        }
        "raw" => {
            let mut c = Command::new("sh");
            c.arg("-c").arg(code);
            c.stdout(std::process::Stdio::piped());
            c.stderr(std::process::Stdio::piped());
            c
        }
        _ => unreachable!(),
    };

    // Set environment variables
    for (key, value) in env_vars {
        cmd.env(key, value);
    }

    // Spawn process
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            send_streaming_error(
                write_tx,
                format!("Failed to execute {} code: {}", language, e),
                "execution_error",
            )
            .await?;
            return Ok(());
        }
    };

    // Get stdout/stderr streams
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();

    // Incremental streaming with 50ms batching and 64KB max buffer (Jan 2026 best practice)
    // - Prevents memory exhaustion on large outputs
    // - Provides real-time feedback (50ms latency, not 1s)
    // - Uses backpressure via bounded channel
    let write_tx_clone = write_tx.clone();
    let streaming_task = tokio::spawn(async move {
        let mut stdout_buffer = String::new();
        let mut stderr_buffer = String::new();
        let mut stdout_bytes = [0u8; 8192]; // 8KB read chunks
        let mut stderr_bytes = [0u8; 8192];
        let mut flush_timer = interval(Duration::from_millis(FLUSH_INTERVAL_MS));
        let mut stdout_done = false;
        let mut stderr_done = false;

        loop {
            tokio::select! {
                // Flush timer (50ms) - send buffered data for real-time feel
                _ = flush_timer.tick() => {
                    let _ = flush_output_buffer(&write_tx_clone, &mut stdout_buffer, "stdout").await;
                    let _ = flush_output_buffer(&write_tx_clone, &mut stderr_buffer, "stderr").await;
                }

                // Read stdout
                result = stdout.read(&mut stdout_bytes), if !stdout_done => {
                    match result {
                        Ok(0) => stdout_done = true,
                        Ok(n) => {
                            // Use lossy conversion to never drop data (replaces invalid UTF-8 with �)
                            stdout_buffer.push_str(&String::from_utf8_lossy(&stdout_bytes[..n]));
                            // Flush immediately if buffer exceeds 64KB (backpressure)
                            if stdout_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                let _ = flush_output_buffer(&write_tx_clone, &mut stdout_buffer, "stdout").await;
                            }
                        }
                        Err(_) => stdout_done = true,
                    }
                }

                // Read stderr
                result = stderr.read(&mut stderr_bytes), if !stderr_done => {
                    match result {
                        Ok(0) => stderr_done = true,
                        Ok(n) => {
                            // Use lossy conversion to never drop data (replaces invalid UTF-8 with �)
                            stderr_buffer.push_str(&String::from_utf8_lossy(&stderr_bytes[..n]));
                            // Flush immediately if buffer exceeds 64KB (backpressure)
                            if stderr_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                let _ = flush_output_buffer(&write_tx_clone, &mut stderr_buffer, "stderr").await;
                            }
                        }
                        Err(_) => stderr_done = true,
                    }
                }
            }

            // Exit when both streams are done
            if stdout_done && stderr_done {
                // Final flush of any remaining data
                let _ = flush_output_buffer(&write_tx_clone, &mut stdout_buffer, "stdout").await;
                let _ = flush_output_buffer(&write_tx_clone, &mut stderr_buffer, "stderr").await;
                break;
            }
        }
    });

    // Wait for process with timeout
    let wait_future = child.wait();
    let wait_result = if timeout > 0 {
        tokio::time::timeout(Duration::from_secs(timeout), wait_future).await
    } else {
        Ok(wait_future.await)
    };

    let status = match wait_result {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            let _ = child.kill().await;
            streaming_task.abort();
            send_streaming_error(
                write_tx,
                format!("Process wait error: {}", e),
                "execution_error",
            )
            .await?;
            return Ok(());
        }
        Err(_) => {
            let _ = child.kill().await;
            streaming_task.abort();
            send_streaming_error(
                write_tx,
                format!("Execution timeout after {}s", timeout),
                "timeout_error",
            )
            .await?;
            return Ok(());
        }
    };

    // Wait for streaming task to complete (captures remaining output after process exit)
    let _ = streaming_task.await;

    let duration_ms = start.elapsed().as_millis() as u64;
    let exit_code = status.code().unwrap_or(-1);

    // Send completion message
    let complete = ExecutionComplete {
        msg_type: "complete".to_string(),
        exit_code,
        execution_time_ms: duration_ms,
    };
    let json = serde_json::to_string(&complete)?;
    let mut response = json.into_bytes();
    response.push(b'\n');
    write_tx
        .send(response)
        .await
        .map_err(|_| "Write queue closed")?;

    Ok(())
}

async fn handle_connection<R, W>(
    mut reader: R,
    write_half: W,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: AsyncBufReadExt + Unpin,
    W: AsyncWriteExt + Unpin + Send + 'static,
{
    // Create bounded channel for writes (decouples read/write paths)
    let (write_tx, mut write_rx) = mpsc::channel::<Vec<u8>>(WRITE_QUEUE_SIZE);

    // Spawn background writer task
    let mut writer = BufWriter::new(write_half);
    let write_task = tokio::spawn(async move {
        while let Some(data) = write_rx.recv().await {
            if let Err(e) = writer.write_all(&data).await {
                eprintln!("Write error: {}", e);
                break;
            }
            if let Err(e) = writer.flush().await {
                eprintln!("Flush error: {}", e);
                break;
            }
        }
        // Graceful shutdown - flush remaining data
        let _ = writer.flush().await;
    });

    // Main loop: read requests, queue responses
    let mut line = String::new();
    loop {
        line.clear();

        // Read request with size limit
        let bytes_read = match reader.read_line(&mut line).await {
            Ok(0) => {
                eprintln!("Connection closed by client");
                break;
            }
            Ok(n) => n,
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        };

        // Validate request size
        if bytes_read > MAX_REQUEST_SIZE_BYTES {
            let _ = send_streaming_error(
                &write_tx,
                format!(
                    "Request too large: {} bytes (max {} bytes)",
                    bytes_read, MAX_REQUEST_SIZE_BYTES
                ),
                "request_error",
            )
            .await;
            continue;
        }

        // Log request for debugging
        eprintln!("Received request ({} bytes)", bytes_read);

        // Parse and execute command
        match serde_json::from_str::<GuestCommand>(&line) {
            Ok(GuestCommand::Ping) => {
                eprintln!("Processing: ping");
                let pong = Pong {
                    msg_type: "pong".to_string(),
                    version: VERSION.to_string(),
                };
                let response_json = serde_json::to_string(&pong)?;
                let mut response = response_json.into_bytes();
                response.push(b'\n');

                if write_tx.send(response).await.is_err() {
                    eprintln!("Write queue closed");
                    break;
                }
            }
            Ok(GuestCommand::InstallPackages { language, packages }) => {
                eprintln!(
                    "Processing: install_packages (language={}, count={})",
                    language,
                    packages.len()
                );
                if install_packages(&language, &packages, &write_tx)
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Ok(GuestCommand::ExecuteCode {
                language,
                code,
                timeout,
                env_vars,
            }) => {
                eprintln!(
                    "Processing: execute_code (language={}, code_size={}, timeout={}s, env_vars={})",
                    language,
                    code.len(),
                    timeout,
                    env_vars.len()
                );
                if execute_code_streaming(&language, &code, timeout, &env_vars, &write_tx)
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Err(e) => {
                eprintln!("JSON parse error: {}", e);
                let _ = send_streaming_error(
                    &write_tx,
                    format!("Invalid command JSON: {}", e),
                    "parse_error",
                )
                .await;
            }
        }
    }

    // Cleanup: signal writer to shutdown
    drop(write_tx); // Closes channel, writer loop exits

    // Wait for writer to finish
    let _ = write_task.await;

    Ok(())
}

async fn listen_virtio_serial() -> Result<(), Box<dyn std::error::Error>> {
    use tokio::fs::OpenOptions;

    // Open dual virtio-serial ports (created by QEMU virtserialport)
    // CMD port: host → guest (read-only)
    // EVENT port: guest → host (write-only)
    eprintln!(
        "Guest agent opening virtio-serial ports: cmd={}, event={}",
        CMD_PORT_PATH, EVENT_PORT_PATH
    );

    loop {
        // Open command port (host → guest, read-only)
        let cmd_file = match OpenOptions::new().read(true).open(CMD_PORT_PATH).await {
            Ok(f) => {
                eprintln!("Guest agent connected to command port (read)");
                f
            }
            Err(e) => {
                eprintln!("Failed to open command port: {}, retrying in 100ms...", e);
                // Reduced from 1s to 100ms: virtio ports available shortly after kernel boot
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            }
        };

        // Open event port (guest → host, write-only)
        let event_file = match OpenOptions::new().write(true).open(EVENT_PORT_PATH).await {
            Ok(f) => {
                eprintln!("Guest agent connected to event port (write)");
                f
            }
            Err(e) => {
                eprintln!("Failed to open event port: {}, retrying in 100ms...", e);
                // Reduced from 1s to 100ms: virtio ports available shortly after kernel boot
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            }
        };

        // Split command port (only need read half)
        let (read_half, _) = tokio::io::split(cmd_file);
        let reader = BufReader::new(read_half);

        // Handle connection with dual ports and error recovery
        if let Err(e) = handle_connection(reader, event_file).await {
            eprintln!("Connection error: {}, reopening ports...", e);
            // Small delay before reconnecting
            tokio::time::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS)).await;
        }
    }
}

/// Reap zombie processes when running as PID 1.
///
/// PID 1 is responsible for reaping orphaned child processes.
/// This async task listens for SIGCHLD signals and calls waitpid()
/// to clean up zombie processes.
///
/// Reference: https://github.com/fpco/pid1-rs
async fn reap_zombies() {
    use tokio::signal::unix::{signal, SignalKind};

    // Create signal stream BEFORE any children are spawned to avoid race conditions
    let mut sigchld = match signal(SignalKind::child()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: Failed to register SIGCHLD handler: {}", e);
            return;
        }
    };

    loop {
        // Wait for SIGCHLD (cancel-safe)
        sigchld.recv().await;

        // Reap all available zombies in a non-blocking loop
        // Multiple children may have exited before we get here
        loop {
            // SAFETY: waitpid with WNOHANG is safe and returns immediately
            let pid = unsafe { libc::waitpid(-1, std::ptr::null_mut(), libc::WNOHANG) };
            match pid {
                p if p > 0 => continue, // Reaped one zombie, check for more
                0 => break,             // No more zombies waiting
                _ => break,             // Error (ECHILD = no children)
            }
        }
    }
}

/// Setup environment when running as PID 1 (init).
///
/// Handles userspace-only initialization after minimal-init.sh:
/// - minimal-init.sh: kernel modules, zram, mounts (has access to initramfs modules)
/// - guest-agent: PATH, env vars, network IP config (userspace only)
///
/// Note: modprobe/insmod won't work here - kernel modules are only in initramfs
/// which is unmounted after switch_root.
fn setup_init_environment() {
    // Set PATH for child processes (uv, python3, bun, etc.)
    std::env::set_var("PATH", "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin");
    eprintln!("Set PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin");

    // Disable uv cache (ephemeral VMs)
    std::env::set_var("UV_NO_CACHE", "1");
    eprintln!("Set UV_NO_CACHE=1");

    // Wait for network interface (up to 1 second, 20ms intervals)
    // virtio_net loaded by minimal-init.sh, eth0 appears shortly after
    for _ in 0..50 {
        if std::path::Path::new("/sys/class/net/eth0").exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // Configure network for gvproxy mode
    // gvproxy gateway at 192.168.127.1 provides DNS
    // Static IP (DHCP unavailable - AF_PACKET not supported)
    if std::path::Path::new("/sys/class/net/eth0").exists() {
        eprintln!("Configuring network...");

        let _ = StdCommand::new("ip")
            .args(["link", "set", "eth0", "up"])
            .status();
        let _ = StdCommand::new("ip")
            .args(["addr", "add", "192.168.127.2/24", "dev", "eth0"])
            .status();
        let _ = StdCommand::new("ip")
            .args(["route", "add", "default", "via", "192.168.127.1"])
            .status();

        eprintln!("Network configured: 192.168.127.2/24 via 192.168.127.1");
    } else {
        eprintln!("Warning: eth0 not found, network unavailable");
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // When running as PID 1, setup environment and zombie reaping
    // This must be done early before any child processes are spawned
    if std::process::id() == 1 {
        eprintln!("Guest agent running as PID 1 (init)...");

        // Setup environment (PATH, UV_NO_CACHE, network)
        setup_init_environment();

        // Enable zombie reaper
        eprintln!("Enabling zombie reaper...");
        tokio::spawn(reap_zombies());
    }

    eprintln!(
        "Guest agent starting (dual ports: cmd={}, event={})...",
        CMD_PORT_PATH, EVENT_PORT_PATH
    );
    listen_virtio_serial().await
}

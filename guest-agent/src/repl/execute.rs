//! Code execution via persistent REPL with streaming output.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio::time::{Duration, interval};

use crate::constants::*;
use crate::error::*;
use crate::repl::spawn::spawn_repl;
use crate::types::GuestResponse;
use crate::validation::{prepend_env_vars, validate_execute_params};

/// Helper to flush a buffer as a Stdout/Stderr response message.
async fn flush_output_buffer(
    write_tx: &mpsc::Sender<Vec<u8>>,
    buffer: &mut String,
    chunk_type: &str,
) -> Result<(), CmdError> {
    if buffer.is_empty() {
        return Ok(());
    }
    let data = std::mem::take(buffer);
    let msg = match chunk_type {
        "stderr" => GuestResponse::Stderr { chunk: data },
        _ => GuestResponse::Stdout { chunk: data },
    };
    send_response(write_tx, &msg).await
}

/// Execute code via persistent REPL with streaming output.
pub(crate) async fn execute_code_streaming(
    language_str: &str,
    code: &str,
    timeout: u64,
    env_vars: &HashMap<String, String>,
    write_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(), CmdError> {
    let language = parse_language(language_str, "supported: python, javascript, raw")?;

    validate_execute_params(code, timeout, env_vars).map_err(CmdError::validation)?;

    let start = Instant::now();

    // Get or create persistent REPL for this language
    let spawn_start = Instant::now();
    let mut was_fresh_spawn = false;
    let mut repl = {
        let mut states = REPL_STATES.lock().await;
        match states.remove(&language) {
            Some(mut existing) => match existing.child.try_wait() {
                Ok(Some(_)) => {
                    log_warn!("REPL for {} died, spawning fresh", language.as_str());
                    was_fresh_spawn = true;
                    spawn_repl(language)
                        .await
                        .map_err(|e| CmdError::Fatal(e.to_string().into()))?
                }
                Ok(None) => existing,
                Err(_) => {
                    was_fresh_spawn = true;
                    spawn_repl(language)
                        .await
                        .map_err(|e| CmdError::Fatal(e.to_string().into()))?
                }
            },
            None => {
                was_fresh_spawn = true;
                spawn_repl(language)
                    .await
                    .map_err(|e| CmdError::Fatal(e.to_string().into()))?
            }
        }
    };
    let spawn_ms = if was_fresh_spawn {
        Some(spawn_start.elapsed().as_millis() as u64)
    } else {
        Some(0)
    };

    // Generate unique sentinel
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let count = SENTINEL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let sentinel_id = format!("{nanos}_{count}");
    let sentinel_prefix = format!("__SENTINEL_{}_", sentinel_id);

    // Prepend env var setup to user code
    let full_code = prepend_env_vars(language, code, env_vars);
    let code_bytes = full_code.as_bytes();

    // Write header + code to REPL stdin
    let header = format!("{} {}\n", sentinel_id, code_bytes.len());
    let write_result = async {
        repl.stdin.write_all(header.as_bytes()).await?;
        repl.stdin.write_all(code_bytes).await?;
        repl.stdin.flush().await
    }
    .await;
    if let Err(e) = write_result {
        log_error!("REPL stdin write failed: {e}");
        let _ = repl.child.kill().await;
        let _ = repl.child.wait().await;
        return Err(CmdError::execution(format!(
            "Failed to send code to REPL: {e}"
        )));
    }

    let process_start = Instant::now();

    // Stream output until sentinel, REPL death, or timeout
    let mut stdout_buffer = String::new();
    let mut stderr_buffer = String::new();
    let mut stderr_line_buf = String::new();
    let mut stdout_bytes = [0u8; 8192];
    let mut stderr_bytes = [0u8; 8192];
    let mut flush_timer = interval(Duration::from_millis(FLUSH_INTERVAL_MS));
    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut sentinel_exit_code: Option<i32> = None;

    let effective_timeout = if timeout > 0 {
        timeout
    } else {
        MAX_TIMEOUT_SECONDS
    };
    let timeout_duration = Duration::from_secs(effective_timeout);

    let loop_result = tokio::time::timeout(timeout_duration, async {
        loop {
            tokio::select! {
                _ = flush_timer.tick() => {
                    let _ = flush_output_buffer(write_tx, &mut stdout_buffer, "stdout").await;
                    let _ = flush_output_buffer(write_tx, &mut stderr_buffer, "stderr").await;
                }

                result = repl.stdout.read(&mut stdout_bytes), if !stdout_done => {
                    match result {
                        Ok(0) => stdout_done = true,
                        Ok(n) => {
                            stdout_buffer.push_str(&String::from_utf8_lossy(&stdout_bytes[..n]));
                            if stdout_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                let _ = flush_output_buffer(write_tx, &mut stdout_buffer, "stdout").await;
                            }
                        }
                        Err(_) => stdout_done = true,
                    }
                }

                result = repl.stderr.read(&mut stderr_bytes), if !stderr_done => {
                    match result {
                        Ok(0) => stderr_done = true,
                        Ok(n) => {
                            let chunk = String::from_utf8_lossy(&stderr_bytes[..n]).into_owned();
                            stderr_line_buf.push_str(&chunk);

                            while let Some(nl) = stderr_line_buf.find('\n') {
                                let line = stderr_line_buf[..nl].to_string();
                                stderr_line_buf = stderr_line_buf[nl + 1..].to_string();

                                if let Some(sentinel_pos) = line.find(&sentinel_prefix) {
                                    let sentinel_part = &line[sentinel_pos..];
                                    if sentinel_part.ends_with("__") {
                                        if sentinel_pos > 0 {
                                            stderr_buffer.push_str(&line[..sentinel_pos]);
                                        }
                                        let code_str = &sentinel_part
                                            [sentinel_prefix.len()..sentinel_part.len() - 2];
                                        sentinel_exit_code =
                                            Some(code_str.parse::<i32>().unwrap_or(-1));
                                        continue;
                                    }
                                }
                                stderr_buffer.push_str(&line);
                                stderr_buffer.push('\n');
                                if stderr_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                    let _ = flush_output_buffer(write_tx, &mut stderr_buffer, "stderr").await;
                                }
                            }
                        }
                        Err(_) => stderr_done = true,
                    }
                }
            }

            if sentinel_exit_code.is_some() {
                // Drain remaining stdout
                loop {
                    match tokio::time::timeout(
                        Duration::from_millis(DRAIN_TIMEOUT_MS),
                        repl.stdout.read(&mut stdout_bytes),
                    )
                    .await
                    {
                        Ok(Ok(0)) | Err(_) => break,
                        Ok(Ok(n)) => {
                            stdout_buffer
                                .push_str(&String::from_utf8_lossy(&stdout_bytes[..n]));
                        }
                        Ok(Err(_)) => break,
                    }
                }
                break;
            }
            if stdout_done && stderr_done {
                break;
            }
        }
    })
    .await;

    // Flush residual buffers
    if !stderr_line_buf.is_empty() {
        stderr_buffer.push_str(&stderr_line_buf);
    }
    let _ = flush_output_buffer(write_tx, &mut stdout_buffer, "stdout").await;
    let _ = flush_output_buffer(write_tx, &mut stderr_buffer, "stderr").await;

    let duration_ms = start.elapsed().as_millis() as u64;
    let process_ms = Some(process_start.elapsed().as_millis() as u64);

    match loop_result {
        Ok(()) if sentinel_exit_code.is_some() => {
            let exit_code = sentinel_exit_code.unwrap();
            REPL_STATES.lock().await.insert(language, repl);

            send_response(
                write_tx,
                &GuestResponse::Complete {
                    exit_code,
                    execution_time_ms: duration_ms,
                    spawn_ms,
                    process_ms,
                },
            )
            .await?;
        }
        Ok(()) => {
            let status = repl.child.wait().await;
            let exit_code = status.map(exit_code_from_status).unwrap_or(-1);
            log_warn!(
                "REPL for {} died with exit_code={}",
                language.as_str(),
                exit_code
            );

            send_response(
                write_tx,
                &GuestResponse::Complete {
                    exit_code,
                    execution_time_ms: duration_ms,
                    spawn_ms,
                    process_ms,
                },
            )
            .await?;
        }
        Err(_) => {
            log_warn!(
                "REPL for {} timed out after {}s, killing",
                language.as_str(),
                effective_timeout
            );
            let _ =
                graceful_terminate_process_group(&mut repl.child, TERM_GRACE_PERIOD_SECONDS).await;

            return Err(CmdError::timeout(format!(
                "Execution timeout after {}s",
                effective_timeout
            )));
        }
    }

    Ok(())
}

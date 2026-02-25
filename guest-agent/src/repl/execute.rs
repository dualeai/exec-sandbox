//! Code execution via persistent REPL with streaming output.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{Duration, interval};

use crate::constants::*;
use crate::error::{
    CmdError, ResponseWriter, exit_code_from_status, graceful_terminate_process_group,
    parse_language,
};
use crate::repl::spawn::spawn_repl;
use crate::types::GuestResponse;
use crate::validation::{prepend_env_vars, validate_execute_params};

/// Helper to flush a buffer as a Stdout/Stderr response message.
async fn flush_output_buffer(
    writer: &ResponseWriter,
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
    writer.send(&msg).await
}

/// Parse a single stderr line for a sentinel marker, returning the exit code if found.
/// Appends any pre-sentinel text to `stderr_buffer`, and non-sentinel lines as-is.
fn parse_stderr_line_for_sentinel(
    line: &str,
    sentinel_prefix: &str,
    stderr_buffer: &mut String,
) -> Option<i32> {
    if let Some(sentinel_pos) = line.find(sentinel_prefix) {
        let sentinel_part = &line[sentinel_pos..];
        if sentinel_part.ends_with("__") {
            if sentinel_pos > 0 {
                stderr_buffer.push_str(&line[..sentinel_pos]);
            }
            let code_str = &sentinel_part[sentinel_prefix.len()..sentinel_part.len() - 2];
            return Some(code_str.parse::<i32>().unwrap_or(-1));
        }
    }
    stderr_buffer.push_str(line);
    stderr_buffer.push('\n');
    None
}

/// Process a raw stderr chunk: append to the line buffer, parse complete lines for sentinel.
/// Returns the sentinel exit code if found in any line.
fn process_stderr_chunk(
    chunk: &[u8],
    stderr_line_buf: &mut String,
    sentinel_prefix: &str,
    stderr_buffer: &mut String,
) -> Option<i32> {
    stderr_line_buf.push_str(&String::from_utf8_lossy(chunk));
    let mut sentinel_exit_code: Option<i32> = None;
    while let Some(nl) = stderr_line_buf.find('\n') {
        let line: String = stderr_line_buf.drain(..=nl).collect();
        // drain includes the '\n'; strip it for sentinel parsing
        let line = &line[..line.len() - 1];
        if let Some(code) = parse_stderr_line_for_sentinel(line, sentinel_prefix, stderr_buffer) {
            // Last sentinel wins if multiple appear in one chunk
            sentinel_exit_code = Some(code);
        }
    }
    sentinel_exit_code
}

/// Drain stdout with a cumulative deadline after sentinel detection or SIGTERM.
/// Uses DRAIN_TIMEOUT_MS per-read to keep latency low while the deadline bounds total time.
async fn drain_stdout(
    stdout: &mut tokio::process::ChildStdout,
    stdout_buffer: &mut String,
    deadline_ms: u64,
) {
    let drain_deadline = Instant::now() + Duration::from_millis(deadline_ms);
    let mut buf = [0u8; 8192];
    loop {
        let remaining = drain_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let per_read = remaining.min(Duration::from_millis(DRAIN_TIMEOUT_MS));
        match tokio::time::timeout(per_read, stdout.read(&mut buf)).await {
            Ok(Ok(0)) | Err(_) | Ok(Err(_)) => break,
            Ok(Ok(n)) => {
                stdout_buffer.push_str(&String::from_utf8_lossy(&buf[..n]));
            }
        }
    }
}

/// Drain both stdout and stderr after SIGTERM, parsing stderr for sentinel.
/// Returns the sentinel exit code if found, or None.
///
/// Once the sentinel is found on stderr, immediately switches to stdout-only
/// drain (same pattern as the normal execution path) to minimize latency.
async fn drain_after_sigterm(
    stdout: &mut tokio::process::ChildStdout,
    stderr: &mut tokio::process::ChildStderr,
    stdout_buffer: &mut String,
    stderr_buffer: &mut String,
    stderr_line_buf: &mut String,
    sentinel_prefix: &str,
) -> Option<i32> {
    let drain_deadline = Instant::now() + Duration::from_millis(200);
    let mut stdout_buf = [0u8; 8192];
    let mut stderr_buf = [0u8; 8192];
    let mut sentinel_exit_code: Option<i32> = None;
    let mut stdout_done = false;
    let mut stderr_done = false;

    loop {
        let remaining = drain_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() || (stdout_done && stderr_done) {
            break;
        }
        let per_read = remaining.min(Duration::from_millis(DRAIN_TIMEOUT_MS));

        // Sentinel found — switch to stdout-only drain (stderr no longer needed)
        if sentinel_exit_code.is_some() {
            drain_stdout(stdout, stdout_buffer, remaining.as_millis() as u64).await;
            break;
        }

        // Once stderr is done, only drain stdout
        if stderr_done {
            match tokio::time::timeout(per_read, stdout.read(&mut stdout_buf)).await {
                Ok(Ok(0)) | Err(_) | Ok(Err(_)) => break,
                Ok(Ok(n)) => {
                    stdout_buffer.push_str(&String::from_utf8_lossy(&stdout_buf[..n]));
                }
            }
            continue;
        }

        // Once stdout is done, only drain stderr
        if stdout_done {
            match tokio::time::timeout(per_read, stderr.read(&mut stderr_buf)).await {
                Ok(Ok(0)) | Err(_) | Ok(Err(_)) => break,
                Ok(Ok(n)) => {
                    if let Some(code) = process_stderr_chunk(
                        &stderr_buf[..n],
                        stderr_line_buf,
                        sentinel_prefix,
                        stderr_buffer,
                    ) {
                        sentinel_exit_code = Some(code);
                    }
                }
            }
            continue;
        }

        // Both pipes active — use select
        tokio::select! {
            result = tokio::time::timeout(per_read, stdout.read(&mut stdout_buf)) => {
                match result {
                    Ok(Ok(0)) | Ok(Err(_)) => { stdout_done = true; }
                    Err(_) => {} // timeout — try again
                    Ok(Ok(n)) => {
                        stdout_buffer.push_str(&String::from_utf8_lossy(&stdout_buf[..n]));
                    }
                }
            }
            result = tokio::time::timeout(per_read, stderr.read(&mut stderr_buf)) => {
                match result {
                    Ok(Ok(0)) | Ok(Err(_)) => { stderr_done = true; }
                    Err(_) => {} // timeout — try again
                    Ok(Ok(n)) => {
                        if let Some(code) = process_stderr_chunk(
                            &stderr_buf[..n],
                            stderr_line_buf,
                            sentinel_prefix,
                            stderr_buffer,
                        ) {
                            sentinel_exit_code = Some(code);
                        }
                    }
                }
            }
        }
    }

    sentinel_exit_code
}

/// Execute code via persistent REPL with streaming output.
pub(crate) async fn execute_code_streaming(
    language_str: &str,
    code: &str,
    timeout: u64,
    env_vars: &HashMap<String, String>,
    writer: &ResponseWriter,
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
    log_info!(
        "[timing] repl_acquired: {}ms (fresh={}, spawn={}ms)",
        crate::monotonic_ms(),
        was_fresh_spawn,
        spawn_ms.unwrap_or(0)
    );

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

    log_info!(
        "[timing] stdin_written: {}ms (code_len={})",
        crate::monotonic_ms(),
        code_bytes.len()
    );

    let process_start = Instant::now();
    let mut first_io_logged = false;

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
                    let _ = flush_output_buffer(writer, &mut stdout_buffer, "stdout").await;
                    let _ = flush_output_buffer(writer, &mut stderr_buffer, "stderr").await;
                }

                result = repl.stdout.read(&mut stdout_bytes), if !stdout_done => {
                    match result {
                        Ok(0) => stdout_done = true,
                        Ok(n) => {
                            if !first_io_logged {
                                first_io_logged = true;
                                log_info!(
                                    "[timing] first_repl_io: {}ms (stdout, {}ms after stdin_write, {} bytes)",
                                    crate::monotonic_ms(),
                                    process_start.elapsed().as_millis(),
                                    n
                                );
                            }
                            stdout_buffer.push_str(&String::from_utf8_lossy(&stdout_bytes[..n]));
                            if stdout_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                let _ = flush_output_buffer(writer, &mut stdout_buffer, "stdout").await;
                            }
                        }
                        Err(_) => stdout_done = true,
                    }
                }

                result = repl.stderr.read(&mut stderr_bytes), if !stderr_done => {
                    match result {
                        Ok(0) => stderr_done = true,
                        Ok(n) => {
                            if !first_io_logged {
                                first_io_logged = true;
                                log_info!(
                                    "[timing] first_repl_io: {}ms (stderr, {}ms after stdin_write, {} bytes)",
                                    crate::monotonic_ms(),
                                    process_start.elapsed().as_millis(),
                                    n
                                );
                            }
                            if let Some(code) = process_stderr_chunk(
                                &stderr_bytes[..n],
                                &mut stderr_line_buf,
                                &sentinel_prefix,
                                &mut stderr_buffer,
                            ) {
                                sentinel_exit_code = Some(code);
                            }
                            if stderr_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                let _ = flush_output_buffer(writer, &mut stderr_buffer, "stderr").await;
                            }
                        }
                        Err(_) => stderr_done = true,
                    }
                }
            }

            if sentinel_exit_code.is_some() {
                drain_stdout(&mut repl.stdout, &mut stdout_buffer, 100).await;
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
    let _ = flush_output_buffer(writer, &mut stdout_buffer, "stdout").await;
    let _ = flush_output_buffer(writer, &mut stderr_buffer, "stderr").await;

    let duration_ms = start.elapsed().as_millis() as u64;
    let process_ms = Some(process_start.elapsed().as_millis() as u64);

    match loop_result {
        Ok(()) if sentinel_exit_code.is_some() => {
            let exit_code = sentinel_exit_code.unwrap();
            REPL_STATES.lock().await.insert(language, repl);

            writer
                .send(&GuestResponse::Complete {
                    exit_code,
                    execution_time_ms: duration_ms,
                    spawn_ms,
                    process_ms,
                })
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

            writer
                .send(&GuestResponse::Complete {
                    exit_code,
                    execution_time_ms: duration_ms,
                    spawn_ms,
                    process_ms,
                })
                .await?;
        }
        Err(_) => {
            log_warn!(
                "REPL for {} timed out after {}s, sending SIGTERM",
                language.as_str(),
                effective_timeout
            );
            let _ =
                graceful_terminate_process_group(&mut repl.child, TERM_GRACE_PERIOD_SECONDS).await;

            // Drain stdout/stderr after SIGTERM — the process may have printed
            // output (e.g. Python SIGTERM handler) and emitted a sentinel.
            let sigterm_exit_code = drain_after_sigterm(
                &mut repl.stdout,
                &mut repl.stderr,
                &mut stdout_buffer,
                &mut stderr_buffer,
                &mut stderr_line_buf,
                &sentinel_prefix,
            )
            .await;

            // If sentinel was found during post-SIGTERM drain, treat as graceful exit
            if let Some(exit_code) = sigterm_exit_code {
                if !stderr_line_buf.is_empty() {
                    stderr_buffer.push_str(&stderr_line_buf);
                }
                let _ = flush_output_buffer(writer, &mut stdout_buffer, "stdout").await;
                let _ = flush_output_buffer(writer, &mut stderr_buffer, "stderr").await;
                writer
                    .send(&GuestResponse::Complete {
                        exit_code,
                        execution_time_ms: duration_ms,
                        spawn_ms,
                        process_ms,
                    })
                    .await?;
                return Ok(());
            }

            return Err(CmdError::timeout(format!(
                "Execution timeout after {}s",
                effective_timeout
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // -------------------------------------------------------------------------
    // parse_stderr_line_for_sentinel — normal cases
    // -------------------------------------------------------------------------

    #[test]
    fn sentinel_basic_exit_zero() {
        let mut buf = String::new();
        let result = parse_stderr_line_for_sentinel(
            "__SENTINEL_123_456_0__",
            "__SENTINEL_123_456_",
            &mut buf,
        );
        assert_eq!(result, Some(0));
        assert!(buf.is_empty(), "no pre-sentinel text expected");
    }

    #[test]
    fn sentinel_nonzero_exit_code() {
        let mut buf = String::new();
        let result =
            parse_stderr_line_for_sentinel("__SENTINEL_1_2_42__", "__SENTINEL_1_2_", &mut buf);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn sentinel_negative_exit_code() {
        let mut buf = String::new();
        let result =
            parse_stderr_line_for_sentinel("__SENTINEL_1_0_-1__", "__SENTINEL_1_0_", &mut buf);
        assert_eq!(result, Some(-1));
    }

    #[test]
    fn sentinel_large_exit_code() {
        let mut buf = String::new();
        let result =
            parse_stderr_line_for_sentinel("__SENTINEL_1_0_137__", "__SENTINEL_1_0_", &mut buf);
        assert_eq!(result, Some(137));
    }

    #[test]
    fn sentinel_with_text_before() {
        let mut buf = String::new();
        let result = parse_stderr_line_for_sentinel(
            "some error output__SENTINEL_1_0_1__",
            "__SENTINEL_1_0_",
            &mut buf,
        );
        assert_eq!(result, Some(1));
        assert_eq!(buf, "some error output");
    }

    // -------------------------------------------------------------------------
    // parse_stderr_line_for_sentinel — non-sentinel lines
    // -------------------------------------------------------------------------

    #[test]
    fn plain_line_buffered_with_newline() {
        let mut buf = String::new();
        let result = parse_stderr_line_for_sentinel("hello world", "__SENTINEL_1_0_", &mut buf);
        assert_eq!(result, None);
        assert_eq!(buf, "hello world\n");
    }

    #[test]
    fn empty_line_buffered() {
        let mut buf = String::new();
        let result = parse_stderr_line_for_sentinel("", "__SENTINEL_1_0_", &mut buf);
        assert_eq!(result, None);
        assert_eq!(buf, "\n");
    }

    // -------------------------------------------------------------------------
    // parse_stderr_line_for_sentinel — edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn sentinel_prefix_without_closing_underscores() {
        let mut buf = String::new();
        let result =
            parse_stderr_line_for_sentinel("__SENTINEL_1_0_42", "__SENTINEL_1_0_", &mut buf);
        // No trailing __ → treated as normal line
        assert_eq!(result, None);
        assert_eq!(buf, "__SENTINEL_1_0_42\n");
    }

    #[test]
    fn sentinel_prefix_with_wrong_suffix() {
        let mut buf = String::new();
        let result = parse_stderr_line_for_sentinel(
            "__SENTINEL_1_0_42_extra__",
            "__SENTINEL_1_0_",
            &mut buf,
        );
        // "42_extra" is not a valid i32, falls back to -1
        assert_eq!(result, Some(-1));
    }

    #[test]
    fn sentinel_non_numeric_exit_code() {
        let mut buf = String::new();
        let result =
            parse_stderr_line_for_sentinel("__SENTINEL_1_0_abc__", "__SENTINEL_1_0_", &mut buf);
        assert_eq!(result, Some(-1));
    }

    #[test]
    fn wrong_sentinel_prefix_ignored() {
        let mut buf = String::new();
        let result =
            parse_stderr_line_for_sentinel("__SENTINEL_999_0_0__", "__SENTINEL_1_0_", &mut buf);
        // Different prefix → not our sentinel
        assert_eq!(result, None);
        assert_eq!(buf, "__SENTINEL_999_0_0__\n");
    }

    #[test]
    fn sentinel_empty_exit_code() {
        let mut buf = String::new();
        let result =
            parse_stderr_line_for_sentinel("__SENTINEL_1_0___", "__SENTINEL_1_0_", &mut buf);
        // Empty string between prefix and __ → parse fails → -1
        assert_eq!(result, Some(-1));
    }

    #[rstest]
    #[case("__SENTINEL_1_0_", "not a sentinel at all")]
    #[case("__SENTINEL_1_0_", "SENTINEL_1_0_5__")]
    #[case("__SENTINEL_1_0_", "__sentinel_1_0_5__")]
    fn sentinel_not_found(#[case] prefix: &str, #[case] line: &str) {
        let mut buf = String::new();
        assert_eq!(parse_stderr_line_for_sentinel(line, prefix, &mut buf), None);
    }

    #[test]
    fn sentinel_accumulates_into_existing_buffer() {
        let mut buf = String::from("previous\n");
        let result = parse_stderr_line_for_sentinel("more stderr", "__SENTINEL_1_0_", &mut buf);
        assert_eq!(result, None);
        assert_eq!(buf, "previous\nmore stderr\n");
    }

    // -------------------------------------------------------------------------
    // process_stderr_chunk — normal cases
    // -------------------------------------------------------------------------

    #[test]
    fn chunk_single_complete_line_with_sentinel() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let chunk = b"__SENTINEL_1_0_0__\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, Some(0));
        assert!(line_buf.is_empty());
        assert!(stderr_buf.is_empty());
    }

    #[test]
    fn chunk_single_plain_line() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let chunk = b"hello world\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert!(line_buf.is_empty(), "complete line should be consumed");
        assert_eq!(stderr_buf, "hello world\n");
    }

    #[test]
    fn chunk_multiple_lines_sentinel_on_last() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let chunk = b"line1\nline2\n__SENTINEL_1_0_5__\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, Some(5));
        assert_eq!(stderr_buf, "line1\nline2\n");
    }

    #[test]
    fn chunk_multiple_lines_sentinel_in_middle() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        // Lines after sentinel are still processed (they become regular stderr)
        let chunk = b"before\n__SENTINEL_1_0_0__\nafter\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, Some(0));
        assert_eq!(stderr_buf, "before\nafter\n");
    }

    // -------------------------------------------------------------------------
    // process_stderr_chunk — partial / multi-chunk delivery
    // -------------------------------------------------------------------------

    #[test]
    fn chunk_partial_line_buffered() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let chunk = b"partial data no newline";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(line_buf, "partial data no newline");
        assert!(stderr_buf.is_empty(), "no complete lines yet");
    }

    #[test]
    fn chunk_split_across_two_deliveries() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let prefix = "__SENTINEL_1_0_";

        // First chunk: partial line
        let r1 = process_stderr_chunk(b"__SENTINEL_1", &mut line_buf, prefix, &mut stderr_buf);
        assert_eq!(r1, None);
        assert_eq!(line_buf, "__SENTINEL_1");

        // Second chunk: completes the line
        let r2 = process_stderr_chunk(b"_0_0__\n", &mut line_buf, prefix, &mut stderr_buf);
        assert_eq!(r2, Some(0));
        assert!(line_buf.is_empty());
    }

    #[test]
    fn chunk_sentinel_split_across_three_deliveries() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let prefix = "__SENTINEL_1_0_";

        let r1 = process_stderr_chunk(b"__SENT", &mut line_buf, prefix, &mut stderr_buf);
        assert_eq!(r1, None);

        let r2 = process_stderr_chunk(b"INEL_1_0_", &mut line_buf, prefix, &mut stderr_buf);
        assert_eq!(r2, None);

        let r3 = process_stderr_chunk(b"99__\n", &mut line_buf, prefix, &mut stderr_buf);
        assert_eq!(r3, Some(99));
    }

    // -------------------------------------------------------------------------
    // process_stderr_chunk — edge / weird cases
    // -------------------------------------------------------------------------

    #[test]
    fn chunk_empty() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let result = process_stderr_chunk(b"", &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert!(line_buf.is_empty());
        assert!(stderr_buf.is_empty());
    }

    #[test]
    fn chunk_only_newlines() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let result =
            process_stderr_chunk(b"\n\n\n", &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "\n\n\n");
    }

    #[test]
    fn chunk_invalid_utf8_lossy() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        // 0xFF is invalid UTF-8, should be replaced with U+FFFD
        let chunk: &[u8] = &[0xFF, b'\n'];
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert!(stderr_buf.contains('\u{FFFD}'));
    }

    #[test]
    fn chunk_very_long_line() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let long_line = "x".repeat(100_000);
        let mut chunk = long_line.clone().into_bytes();
        chunk.push(b'\n');
        let result =
            process_stderr_chunk(&chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf.len(), 100_001); // long_line + \n
    }

    #[test]
    fn chunk_sentinel_embedded_in_long_output() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let prefix = "__SENTINEL_1_0_";
        let mut chunk = Vec::new();
        for i in 0..1000 {
            chunk.extend_from_slice(format!("line {i}\n").as_bytes());
        }
        chunk.extend_from_slice(b"__SENTINEL_1_0_0__\n");
        for i in 0..100 {
            chunk.extend_from_slice(format!("post {i}\n").as_bytes());
        }
        let result = process_stderr_chunk(&chunk, &mut line_buf, prefix, &mut stderr_buf);
        assert_eq!(result, Some(0));
        // All 1000 pre-sentinel + 100 post-sentinel lines should be in stderr_buf
        assert_eq!(stderr_buf.matches('\n').count(), 1100);
    }

    #[test]
    fn chunk_preserves_line_buf_across_calls() {
        let mut line_buf = String::from("leftover");
        let mut stderr_buf = String::new();
        let result = process_stderr_chunk(
            b" data\n",
            &mut line_buf,
            "__SENTINEL_1_0_",
            &mut stderr_buf,
        );
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "leftover data\n");
        assert!(line_buf.is_empty());
    }

    #[test]
    fn chunk_mixed_complete_and_partial() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let chunk = b"complete\npartial";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "complete\n");
        assert_eq!(line_buf, "partial");
    }

    // -------------------------------------------------------------------------
    // process_stderr_chunk — out-of-bound / adversarial cases
    // -------------------------------------------------------------------------

    #[test]
    fn chunk_fake_sentinel_wrong_id() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let chunk = b"__SENTINEL_999_999_0__\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "__SENTINEL_999_999_0__\n");
    }

    #[test]
    fn chunk_sentinel_like_pattern_incomplete() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        // Looks like sentinel but missing trailing __
        let chunk = b"__SENTINEL_1_0_42\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "__SENTINEL_1_0_42\n");
    }

    #[test]
    fn chunk_multiple_sentinels_last_wins() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let chunk = b"__SENTINEL_1_0_1__\n__SENTINEL_1_0_2__\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        // Last exit code wins (loop continues after first sentinel)
        assert_eq!(result, Some(2));
    }

    #[test]
    fn chunk_sentinel_with_binary_noise_before() {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let mut chunk: Vec<u8> = vec![0x00, 0x01, 0x02];
        chunk.extend_from_slice(b"__SENTINEL_1_0_7__\n");
        let result =
            process_stderr_chunk(&chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, Some(7));
    }

    #[rstest]
    #[case(i32::MAX)]
    #[case(i32::MIN)]
    fn chunk_sentinel_boundary_exit_codes(#[case] code: i32) {
        let mut line_buf = String::new();
        let mut stderr_buf = String::new();
        let prefix = "__SENTINEL_1_0_";
        let line = format!("{prefix}{code}__\n");
        let result = process_stderr_chunk(line.as_bytes(), &mut line_buf, prefix, &mut stderr_buf);
        assert_eq!(result, Some(code));
    }
}

//! Code execution via persistent REPL with streaming output.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::time::{Duration, interval};

use crate::constants::*;
use crate::error::{
    CmdError, ResponseWriter, exit_code_from_status, graceful_terminate_process_group,
    wait_child_bounded,
};
use crate::repl::spawn::{ReplState, spawn_repl};
use crate::types::{GuestResponse, Language};
use crate::validation::{prepend_env_vars, validate_code_params, validate_env_vars};

/// The idle REPL of each language, between execs. Persists across host connections.
/// The agent serves one command at a time: a REPL is out of it while one runs.
static REPL_STATES: LazyLock<std::sync::Mutex<HashMap<Language, ReplState>>> =
    LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// REPL_STATES, held for one statement (never across an await).
fn idle_repls() -> std::sync::MutexGuard<'static, HashMap<Language, ReplState>> {
    // A panic while held cannot leave the map half written: use it as it is.
    REPL_STATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Monotonic counter for sentinel IDs. Combined with a nanosecond timestamp,
/// it makes each ID unique without the uuid crate.
static SENTINEL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Read at most `limit` bytes of a /proc or /sys file for a diagnostic log line.
fn read_bounded_diagnostic(path: &str, limit: u64) -> String {
    use std::io::Read as _;

    let Ok(file) = std::fs::File::open(path) else {
        return "unavailable".to_string();
    };
    let mut value = String::new();
    match file.take(limit).read_to_string(&mut value) {
        Ok(_) => value.trim().to_string(),
        Err(error) => format!("read_error:{error}"),
    }
}

/// Capture enough bounded state to distinguish an interpreter/read stall from
/// a normal long-running program on the next failure. Never delay termination
/// on an external diagnostic command.
fn log_repl_timeout_diagnostics(pid: Option<u32>, first_io_logged: bool) {
    let Some(pid) = pid else {
        log_warn!("[repl-timeout] pid=unavailable first_io={first_io_logged}");
        return;
    };

    let status = read_bounded_diagnostic(&format!("/proc/{pid}/status"), 8 * 1024);
    let status_summary = status
        .lines()
        .filter(|line| {
            line.starts_with("State:")
                || line.starts_with("VmRSS:")
                || line.starts_with("VmSwap:")
                || line.starts_with("Threads:")
                || line.starts_with("voluntary_ctxt_switches:")
                || line.starts_with("nonvoluntary_ctxt_switches:")
        })
        .collect::<Vec<_>>()
        .join("; ");
    let wchan = read_bounded_diagnostic(&format!("/proc/{pid}/wchan"), 256);
    let stack =
        read_bounded_diagnostic(&format!("/proc/{pid}/stack"), 4 * 1024).replace('\n', "; ");
    let memory_psi = read_bounded_diagnostic("/proc/pressure/memory", 2 * 1024).replace('\n', "; ");
    let zram_mm = read_bounded_diagnostic("/sys/block/zram0/mm_stat", 2 * 1024);
    let members = group_members_diagnostic(pid);

    log_warn!(
        "[repl-timeout] pid={pid} first_io={first_io_logged} status=[{status_summary}] wchan={wchan} stack=[{stack}] members=[{members}] memory_psi=[{memory_psi}] zram_mm=[{zram_mm}]"
    );
}

/// Group members a `[repl-timeout]` line describes; the others are counted.
const DIAGNOSED_MEMBERS: usize = 8;

/// The other processes of the REPL's group (the REPL leads it): pid, name, state, wait
/// channel and kernel stack of each. A REPL that runs an exec waits for its children, so
/// the member where the exec waits tells a stalled command from a slow one.
fn group_members_diagnostic(leader: u32) -> String {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return "unavailable".to_string();
    };
    let mut members = Vec::new();
    let mut more = 0usize;
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == leader {
            continue;
        }
        let stat = read_bounded_diagnostic(&format!("/proc/{pid}/stat"), 1024);
        let Some((comm, state)) = group_member(&stat, leader) else {
            continue;
        };
        if members.len() == DIAGNOSED_MEMBERS {
            more += 1;
            continue;
        }
        let wchan = read_bounded_diagnostic(&format!("/proc/{pid}/wchan"), 256);
        let stack = read_bounded_diagnostic(&format!("/proc/{pid}/stack"), 512).replace('\n', "; ");
        members.push(format!(
            "{pid} {comm} {state} wchan={wchan} stack=[{stack}]"
        ));
    }
    if more > 0 {
        members.push(format!("{more} more"));
    }
    members.join(" | ")
}

/// The name and state of the process whose /proc/<pid>/stat is `stat`, when its process
/// group is `pgid`. The name can hold spaces and parentheses: the fields after it start
/// after its last ')'. Its control characters become '?': the code names its processes
/// (prctl), and a newline would start a line of its own in the log.
fn group_member(stat: &str, pgid: u32) -> Option<(String, &str)> {
    let (head, rest) = stat.rsplit_once(')')?;
    let (_, comm) = head.split_once('(')?;
    let mut fields = rest.split_whitespace(); // state ppid pgrp ...
    let state = fields.next()?;
    let pgrp = fields.nth(1)?.parse::<u32>().ok()?;
    let comm = comm.replace(|c: char| c.is_control(), "?");
    (pgrp == pgid).then_some((comm, state))
}

/// The frame of a piece of stdout.
fn stdout_frame(chunk: String) -> GuestResponse {
    GuestResponse::Stdout { chunk }
}

/// The frame of a piece of stderr.
fn stderr_frame(chunk: String) -> GuestResponse {
    GuestResponse::Stderr { chunk }
}

/// Send `buffer` in `frame` messages (empty after Ok). Cancel-safe: a piece leaves the buffer only
/// once its queue slot is reserved, so a cancel (the exec's deadline) loses no text, and what
/// is left goes out with the flush after the loop.
async fn flush_output_buffer(
    writer: &ResponseWriter,
    buffer: &mut String,
    frame: fn(String) -> GuestResponse,
) -> Result<(), CmdError> {
    while !buffer.is_empty() {
        let permit = writer.reserve().await?;
        // At most MAX_BUFFER_SIZE_BYTES of text per frame, cut on a character boundary: JSON
        // writes a control character as \u00XX (6 bytes), so each frame stays below the host's
        // line limit (guest_channel.STREAM_BUFFER_LIMIT, 512 KB) whatever the output holds. A
        // buffer can pass that size: a stderr line before the sentinel, the stdout drained after it.
        let piece = if buffer.len() <= MAX_BUFFER_SIZE_BYTES {
            std::mem::take(buffer) // one frame: no copy
        } else {
            let mut end = MAX_BUFFER_SIZE_BYTES;
            while !buffer.is_char_boundary(end) {
                end -= 1;
            }
            let piece = buffer[..end].to_owned();
            buffer.replace_range(..end, ""); // one move of the rest, no walk of its characters
            piece
        };
        permit.send(&frame(piece))?;
    }
    Ok(())
}

/// Send both output buffers, stdout first: they hold only output that may go out.
/// A send error is left to the next send, which meets it too.
async fn flush_both(
    writer: &ResponseWriter,
    stdout_buffer: &mut String,
    stderr_buffer: &mut String,
) {
    let _ = flush_output_buffer(writer, stdout_buffer, stdout_frame).await;
    let _ = flush_output_buffer(writer, stderr_buffer, stderr_frame).await;
}

/// True when nothing was sent in the last FLUSH_INTERVAL_MS: the first output
/// after a quiet time then goes out at once, not at the next flush tick (up to
/// FLUSH_INTERVAL_MS later). A 1 ms wait, to send a short exec's output with its
/// end, is no faster and makes the first output 1 ms late.
fn flush_due(last_flush: Option<Instant>) -> bool {
    last_flush.is_none_or(|at| at.elapsed() >= Duration::from_millis(FLUSH_INTERVAL_MS))
}

/// Parse a single stderr line for a sentinel marker, returning the exit code if found.
/// The code is kept like a process exit status (0-255, 255 if it is not a number):
/// the host reads -1 as a timeout only.
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
            return Some(code_str.parse::<i32>().map_or(255, |code| code & 0xFF));
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
    stderr_line_buf: &mut Vec<u8>,
    sentinel_prefix: &str,
    stderr_buffer: &mut String,
) -> Option<i32> {
    // Bytes until a line ends, decoded line by line: '\n' is never inside a multi-byte
    // UTF-8 character, so a character that a pipe read split stays whole in the partial line
    stderr_line_buf.extend_from_slice(chunk);

    // Single linear pass: scan complete lines in place, then drain them all at
    // once. A drain per line would cost O(remaining) each time — quadratic on
    // newline-dense chunks.
    let mut sentinel_exit_code: Option<i32> = None;
    let mut consumed = 0;
    for line in stderr_line_buf.split_inclusive(|&byte| byte == b'\n') {
        if line.last() != Some(&b'\n') {
            break; // trailing partial line stays buffered for the next chunk
        }
        consumed += line.len();
        let text = String::from_utf8_lossy(&line[..line.len() - 1]);
        if let Some(code) = parse_stderr_line_for_sentinel(&text, sentinel_prefix, stderr_buffer) {
            // Last sentinel wins if multiple appear in one chunk
            sentinel_exit_code = Some(code);
        }
    }
    stderr_line_buf.drain(..consumed);

    // Bound the un-terminated tail. A newline-free flood would otherwise grow
    // this buffer without limit — the caller's MAX_STDERR_BYTES cap clears
    // stderr_buffer but keeps reading here for the sentinel. A sentinel line is
    // short and always ends in '\n', so retaining only the final window cannot
    // drop one; any dropped prefix is already-over-limit data.
    let max_tail = MAX_STDERR_BYTES + sentinel_prefix.len();
    if stderr_line_buf.len() > max_tail {
        let cut = stderr_line_buf.len() - max_tail;
        // From the next character start: no continuation byte begins the kept window
        let boundary = (cut..stderr_line_buf.len())
            .find(|&i| stderr_line_buf[i] & 0b1100_0000 != 0b1000_0000)
            .unwrap_or(stderr_line_buf.len());
        stderr_line_buf.drain(..boundary);
    }

    sentinel_exit_code
}

/// The end of one exec's stdout. Each REPL writes `__STDOUT_END_<id>__\n` to
/// stdout after flushing it and before the stderr sentinel, so the agent stops
/// reading stdout at the marker instead of waiting for the pipe to go idle.
/// The marker is removed from the output. A tail that may be the start of a
/// marker split across reads is held back until the next chunk shows it is not,
/// and so is an incomplete UTF-8 character at the end of the output (a pipe read
/// can end inside one): each output decodes alone.
struct StdoutEnd {
    marker: Vec<u8>,
    pending: Vec<u8>,
    seen: bool,
    utf8_tail: Vec<u8>,
}

impl StdoutEnd {
    fn new(sentinel_id: &str) -> Self {
        Self {
            marker: format!("__STDOUT_END_{sentinel_id}__\n").into_bytes(),
            pending: Vec::new(),
            seen: false,
            utf8_tail: Vec::new(),
        }
    }

    /// The output bytes of `chunk` that are known not to be part of the marker, up to
    /// the last whole UTF-8 character. Bytes after the marker (a background child of the
    /// exec) stay output.
    fn feed(&mut self, chunk: &[u8]) -> Vec<u8> {
        let output = self.marker_free(chunk);
        let mut bytes = std::mem::take(&mut self.utf8_tail);
        bytes.extend_from_slice(&output);
        self.utf8_tail = bytes.split_off(bytes.len() - incomplete_utf8_tail(&bytes));
        bytes
    }

    fn marker_free(&mut self, chunk: &[u8]) -> Vec<u8> {
        if self.seen {
            return chunk.to_vec();
        }
        self.pending.extend_from_slice(chunk);
        if let Some(pos) = self
            .pending
            .windows(self.marker.len())
            .position(|window| window == self.marker.as_slice())
        {
            self.seen = true;
            let mut output = std::mem::take(&mut self.pending);
            output.drain(pos..pos + self.marker.len());
            return output;
        }
        // Longest tail of `pending` that starts the marker: wait for more bytes.
        let held = (1..self.marker.len())
            .rev()
            .find(|&k| self.pending.ends_with(&self.marker[..k]))
            .unwrap_or(0);
        let rest = self.pending.split_off(self.pending.len() - held);
        std::mem::replace(&mut self.pending, rest)
    }

    /// The held-back bytes: an incomplete character, then a tail that may start the marker.
    fn held(&self) -> Vec<u8> {
        [self.utf8_tail.as_slice(), self.pending.as_slice()].concat()
    }

    /// The held-back bytes, taken: the end of the output when no marker came (user code
    /// closed or redirected fd 1).
    fn finish(&mut self) -> Vec<u8> {
        let rest = self.held();
        self.utf8_tail.clear();
        self.pending.clear();
        rest
    }
}

/// Length of an incomplete UTF-8 character at the end of `bytes` (0 to 3): a lead byte whose
/// continuation bytes have not come yet.
fn incomplete_utf8_tail(bytes: &[u8]) -> usize {
    for back in 1..=bytes.len().min(3) {
        let byte = bytes[bytes.len() - back];
        if byte & 0b1100_0000 == 0b1000_0000 {
            continue; // a continuation byte: its lead byte is further back
        }
        let width = match byte {
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => return 0, // ASCII or a byte no character starts with: nothing to wait for
        };
        return if width > back { back } else { 0 };
    }
    0
}

/// `bytes` without an incomplete UTF-8 character at its end (incomplete_utf8_tail).
fn whole_characters(bytes: &[u8]) -> &[u8] {
    &bytes[..bytes.len() - incomplete_utf8_tail(bytes)]
}

/// Fallback after the sentinel when the stdout marker has not come: stop once
/// stdout is idle for this long (user code that closed or redirected fd 1).
///
/// A single continuous idle window (rather than N discrete short windows)
/// tolerates event-loop overhead under QEMU TCG (~8× slower than KVM/HVF),
/// where a byte arriving late in a short window could otherwise trip a false
/// timeout boundary while data is still in the pipe buffer.
const DRAIN_IDLE_MS: u64 = 40;

/// Count `output` (stdout after its end marker is removed) into `stdout_total`, and
/// keep it in `stdout_buffer` while the total stays at MAX_STDOUT_BYTES or under: the cap
/// rule of all stdout that may go out (a read that crosses the cap is dropped).
fn keep_under_stdout_cap(output: &[u8], stdout_total: &mut usize, stdout_buffer: &mut String) {
    *stdout_total += output.len();
    if *stdout_total <= MAX_STDOUT_BYTES {
        stdout_buffer.push_str(&String::from_utf8_lossy(output));
    }
}

/// Read stdout until its end marker (`end`), an idle window of DRAIN_IDLE_MS,
/// or the cumulative `deadline_ms`, whichever comes first, counting each read
/// into `stdout_total` (keep_under_stdout_cap). The held-back tail stays in
/// `end`: the caller adds it (`end.finish()`) when it keeps the output.
async fn drain_stdout(
    stdout: &mut tokio::process::ChildStdout,
    end: &mut StdoutEnd,
    stdout_buffer: &mut String,
    stdout_total: &mut usize,
    deadline_ms: u64,
) {
    let total_deadline = Instant::now() + Duration::from_millis(deadline_ms);
    let mut buf = [0u8; OUTPUT_READ_BYTES];
    while !end.seen {
        let total_remaining = total_deadline.saturating_duration_since(Instant::now());
        if total_remaining.is_zero() {
            break;
        }
        let window = total_remaining.min(Duration::from_millis(DRAIN_IDLE_MS));
        match tokio::time::timeout(window, stdout.read(&mut buf)).await {
            Ok(Ok(n)) if n > 0 => {
                keep_under_stdout_cap(&end.feed(&buf[..n]), stdout_total, stdout_buffer);
            }
            // EOF, read error, idle window elapsed, or total deadline hit.
            _ => break,
        }
    }
}

/// How the read loop of an exec ended. A sentinel read wins over a deadline that cut the
/// loop after it (inside a flush): the code ran to its end, and its REPL can serve again.
#[derive(Debug, PartialEq)]
enum LoopEnd {
    Sentinel(i32),
    /// Both pipes ended with no sentinel: the REPL is gone
    ReplDied,
    Timeout,
}

impl LoopEnd {
    fn of(timed_out: bool, sentinel_exit_code: Option<i32>) -> Self {
        match (sentinel_exit_code, timed_out) {
            (Some(exit_code), _) => Self::Sentinel(exit_code),
            (None, false) => Self::ReplDied,
            (None, true) => Self::Timeout,
        }
    }
}

/// The output of one exec after its read loop: the unsent buffers, the stdout end marker and
/// the byte totals. Every way an exec ends settles it with the loop's rules: a stream past its
/// cap (MAX_STDOUT_BYTES, MAX_STDERR_BYTES) keeps nothing more, and ends the exec with
/// OUTPUT_LIMIT unless the exec timed out or the VM must retire (137); stdout goes out up to
/// its cap (its held tail counted), stderr up to its cap and the lines of the read that
/// crosses it. The buffers hold only output that may go out: the cap rules apply as text
/// enters a buffer, before any await (stderr: the lines past the cap leave it in the same
/// step), and every flush sends what they hold (also the text that a deadline left
/// there, cutting a flush). A stream is capped when its total is past its cap: no flag to keep
/// in step.
struct ExecOutput {
    stdout_buffer: String,
    stdout_end: StdoutEnd,
    stdout_total: usize,
    stderr_buffer: String,
    stderr_line_buf: Vec<u8>,
    stderr_total: usize,
}

impl ExecOutput {
    fn stdout_capped(&self) -> bool {
        self.stdout_total > MAX_STDOUT_BYTES
    }

    fn stderr_capped(&self) -> bool {
        self.stderr_total > MAX_STDERR_BYTES
    }

    /// Read the stdout that trails the sentinel (drain_stdout, at most `deadline_ms`). Most
    /// execs read the stdout marker before the sentinel; under TCG (~8x slower) or after a JS
    /// back-pressure yield, stdout can trail it. The reads past the cap are dropped
    /// (keep_under_stdout_cap); flush_rest sends what was kept.
    async fn drain_trailing_stdout(
        &mut self,
        stdout: &mut tokio::process::ChildStdout,
        deadline_ms: u64,
    ) {
        drain_stdout(
            stdout,
            &mut self.stdout_end,
            &mut self.stdout_buffer,
            &mut self.stdout_total,
            deadline_ms,
        )
        .await;
    }

    /// The flush after the loop, by how it ended (`end`). An exec that ended (a sentinel or the
    /// REPL's death) adds the held stdout tail, counted (keep_under_stdout_cap), and the
    /// partial stderr line of a stream at its cap or under. After a timeout both stay for the
    /// drain (finish_after_sigterm): when the sentinel follows, a marker, a sentinel line or a
    /// character split at the deadline comes out whole. Then both buffers go out.
    async fn flush_rest(&mut self, writer: &ResponseWriter, end: &LoopEnd) {
        if *end != LoopEnd::Timeout {
            let tail = self.stdout_end.finish();
            keep_under_stdout_cap(&tail, &mut self.stdout_total, &mut self.stdout_buffer);
            if !self.stderr_capped() {
                let partial = std::mem::take(&mut self.stderr_line_buf);
                self.stderr_buffer
                    .push_str(&String::from_utf8_lossy(&partial));
            }
        }
        self.flush(writer).await;
    }

    /// Send both buffers: they hold only output that may go out.
    async fn flush(&mut self, writer: &ResponseWriter) {
        flush_both(writer, &mut self.stdout_buffer, &mut self.stderr_buffer).await;
    }

    /// After a timeout's graceful_terminate_process_group: drain_after_sigterm. When the drain
    /// reads the sentinel (the exec ended at the deadline, or its code handled SIGTERM), what
    /// was kept goes out (a stream capped before the drain keeps nothing more). With none, only
    /// the text held at the deadline goes out (held_at_deadline), not the drain's reads; the
    /// caller then ends the exec (the timeout error, or 137 when the VM must retire). Returns
    /// the sentinel exit code if found.
    async fn finish_after_sigterm(
        &mut self,
        writer: &ResponseWriter,
        stdout: &mut tokio::process::ChildStdout,
        stderr: &mut tokio::process::ChildStderr,
        sentinel_prefix: &str,
    ) -> Option<i32> {
        let (mut stdout_held, mut stderr_held) = self.held_at_deadline();
        let sentinel_exit_code = self
            .drain_after_sigterm(stdout, stderr, sentinel_prefix)
            .await;
        if sentinel_exit_code.is_some() {
            self.flush(writer).await;
        } else {
            flush_both(writer, &mut stdout_held, &mut stderr_held).await;
        }
        sentinel_exit_code
    }

    /// The text read before a timeout's deadline that waits for the drain to end it: stdout's
    /// held tail and the partial stderr line, each under the cap rules (the tail counted with
    /// keep_under_stdout_cap; the line goes while stderr is at its cap or under) and without an
    /// incomplete last character (its rest came after the deadline). With no sentinel after
    /// the drain, it is the end of the output (a progress line with no newline); a tail that
    /// starts the marker is then user output, as the REPL writes the marker just before its
    /// sentinel.
    fn held_at_deadline(&self) -> (String, String) {
        let mut stdout_total = self.stdout_total;
        let mut stdout_held = String::new();
        keep_under_stdout_cap(
            whole_characters(&self.stdout_end.held()),
            &mut stdout_total,
            &mut stdout_held,
        );
        let stderr_held = if self.stderr_capped() {
            String::new()
        } else {
            String::from_utf8_lossy(whole_characters(&self.stderr_line_buf)).into_owned()
        };
        (stdout_held, stderr_held)
    }

    /// Read the output left after a timeout's graceful_terminate_process_group, at most
    /// 200 ms: stdout to its end, stderr to its end or the sentinel (a REPL that handled
    /// SIGTERM writes it). The group has ended, so both pipes end at once, unless a process
    /// that left the group holds one. Each read counts with the loop's rules: stdout past
    /// MAX_STDOUT_BYTES is dropped (keep_under_stdout_cap, the held tail too); stderr keeps
    /// the lines up to and with the read that crosses MAX_STDERR_BYTES, then nothing, and the
    /// partial line is kept while stderr is at its cap or under. Returns the sentinel exit
    /// code if found.
    async fn drain_after_sigterm(
        &mut self,
        stdout: &mut tokio::process::ChildStdout,
        stderr: &mut tokio::process::ChildStderr,
        sentinel_prefix: &str,
    ) -> Option<i32> {
        let Self {
            stdout_buffer,
            stdout_end,
            stdout_total,
            stderr_buffer,
            stderr_line_buf,
            stderr_total,
        } = self;
        let mut sentinel_exit_code: Option<i32> = None;
        let stdout_part = async {
            let mut buf = [0u8; OUTPUT_READ_BYTES];
            while let Ok(n @ 1..) = stdout.read(&mut buf).await {
                keep_under_stdout_cap(&stdout_end.feed(&buf[..n]), stdout_total, stdout_buffer);
            }
        };
        let stderr_part = async {
            let mut buf = [0u8; OUTPUT_READ_BYTES];
            // The buffer's length after the read that crossed the cap; already past it at the
            // start, nothing more is kept (the buffer holds only output that may go out)
            let mut kept = (*stderr_total > MAX_STDERR_BYTES).then_some(stderr_buffer.len());
            while sentinel_exit_code.is_none()
                && let Ok(n @ 1..) = stderr.read(&mut buf).await
            {
                sentinel_exit_code = process_stderr_chunk(
                    &buf[..n],
                    stderr_line_buf,
                    sentinel_prefix,
                    stderr_buffer,
                );
                *stderr_total += n;
                if let Some(len) = kept {
                    stderr_buffer.truncate(len);
                } else if *stderr_total > MAX_STDERR_BYTES {
                    kept = Some(stderr_buffer.len());
                }
            }
        };
        let _ = tokio::time::timeout(Duration::from_millis(200), async {
            tokio::join!(stdout_part, stderr_part)
        })
        .await;

        keep_under_stdout_cap(&stdout_end.finish(), stdout_total, stdout_buffer);
        if *stderr_total <= MAX_STDERR_BYTES {
            stderr_buffer.push_str(&String::from_utf8_lossy(&std::mem::take(stderr_line_buf)));
        }
        sentinel_exit_code
    }

    /// The result of an exec whose REPL died with `exit_code`: output past a cap is
    /// OUTPUT_LIMIT here too (a new REPL serves the next exec), unless the VM must retire
    /// (137, which the host must see).
    fn death_result(&self, exit_code: i32) -> Result<i32, CmdError> {
        match self.limit_message() {
            Some(message) if exit_code != 137 => Err(CmdError::output_limit(format!(
                "{message}; the REPL then exited with code {exit_code}"
            ))),
            _ => Ok(exit_code),
        }
    }

    /// The OUTPUT_LIMIT text of the capped streams, or None.
    fn limit_message(&self) -> Option<String> {
        let mut parts = Vec::new();
        if self.stdout_capped() {
            parts.push(format!(
                "stdout {} bytes exceeds {} limit",
                self.stdout_total, MAX_STDOUT_BYTES
            ));
        }
        if self.stderr_capped() {
            parts.push(format!(
                "stderr {} bytes exceeds {} limit",
                self.stderr_total, MAX_STDERR_BYTES
            ));
        }
        (!parts.is_empty()).then(|| parts.join(", "))
    }
}

/// Generate a unique sentinel ID and its prefix for matching on stderr.
fn generate_sentinel() -> (String, String) {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let count = SENTINEL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let id = format!("{nanos}_{count}");
    let prefix = format!("__SENTINEL_{}_", id);
    (id, prefix)
}

/// Write a code submission (header + body) to the REPL's stdin. `reseed`: the REPL
/// reseeds its own random generators from the kernel before the code runs.
async fn submit_code_to_repl<W: AsyncWrite + Unpin>(
    stdin: &mut W,
    sentinel_id: &str,
    code: &[u8],
    reseed: bool,
) -> std::io::Result<()> {
    let header = format!("{sentinel_id} {} {}\n", code.len(), u8::from(reseed));
    stdin.write_all(header.as_bytes()).await?;
    stdin.write_all(code).await?;
    stdin.flush().await
}

/// The live REPL of `language`, taken out of REPL_STATES, or a new one when none
/// runs or it died while idle (true: new). A new REPL is returned once it has written its
/// ready line, within `ready_timeout`; one that exits or stalls before is killed, and the
/// error says the REPL did not start (the caller's code never ran). The caller puts it
/// back when done. The agent serves one command at a time: two callers never start two REPLs.
async fn take_or_spawn_repl(
    language: Language,
    ready_timeout: Duration,
) -> Result<(ReplState, bool), Box<dyn std::error::Error>> {
    let idle = idle_repls().remove(&language);
    let alive = match idle {
        Some(mut existing) => match existing.child.try_wait() {
            Ok(None) => Some(existing),
            Ok(Some(status)) => {
                // An exit 137 retires the VM: spawn_repl then refuses.
                let exit_code =
                    crate::oom_guard::reconcile_retirement_exit(exit_code_from_status(status));
                log_warn!(
                    "REPL for {} died while idle with exit_code={exit_code}",
                    language.as_str()
                );
                None
            }
            Err(_) => None,
        },
        None => None,
    };
    if let Some(existing) = alive {
        return Ok((existing, false));
    }
    let mut repl = spawn_repl(language)?;
    let failure = match tokio::time::timeout(ready_timeout, read_ready_line(&mut repl.stderr)).await
    {
        Ok(Ok(before)) => {
            if !before.is_empty() {
                log_warn!(
                    "REPL for {} wrote before its ready line: {}",
                    language.as_str(),
                    before.trim_end()
                );
            }
            return Ok((repl, true));
        }
        Ok(Err(stderr)) => format!("REPL exited at start: {}", stderr.trim_end()),
        Err(_) => format!("REPL did not start within {}ms", ready_timeout.as_millis()),
    };
    kill_repl(&mut repl).await;
    Err(failure.into())
}

/// End a REPL that is not put back in REPL_STATES: SIGKILL, then a bounded wait.
async fn kill_repl(repl: &mut ReplState) {
    let _ = repl.child.start_kill();
    let _ = wait_child_bounded(&mut repl.child, "SIGKILL", KILL_WAIT).await;
}

/// Warm up the REPL of `language`: the live one when one runs (an L1 restore, a
/// warm-up before), else a new one; run a no-op in it, and keep it for the next
/// exec. Err holds the message of the warm-up ack; a REPL that fails the no-op
/// is killed.
pub(crate) async fn warm_repl(language: Language) -> Result<(), String> {
    let (mut repl, _) = take_or_spawn_repl(language, WARM_UP_TIMEOUT)
        .await
        .map_err(|e| {
            log_warn!("Eager REPL spawn failed: {e}");
            e.to_string()
        })?;
    // A new REPL has written its ready line. The no-op loads the modules that the
    // REPL loads only on demand and starts its output paths (on a live REPL it also
    // maps its pages again), so the first exec of a user is not the REPL's first one.
    let warm_start = Instant::now();
    match warm_exercise_repl(&mut repl, language).await {
        Ok(()) => {
            let warm_ms = warm_start.elapsed().as_millis();
            idle_repls().insert(language, repl);
            log_info!("REPL pre-warmed for {} ({warm_ms}ms)", language.as_str());
            Ok(())
        }
        Err(e) => {
            log_warn!("REPL warm-up failed for {}: {e}", language.as_str());
            kill_repl(&mut repl).await;
            Err(format!("warm-up failed: {e}"))
        }
    }
}

/// The line each REPL writes to its stderr right before it reads its first exec.
const READY_LINE: &[u8] = b"__REPL_READY__\n";

/// The time a warm-up gives a REPL to start, and then again to run its no-op (before
/// the VM is advertised: generous, for TCG under load).
const WARM_UP_TIMEOUT: Duration = Duration::from_secs(120);

/// Read a new REPL's stderr up to its ready line, one byte at a time: nothing after the
/// line is taken from the pipe. Ok(what came before it: interpreter warnings), or Err(the
/// stderr) when the pipe closed first (the REPL failed at start). Both keep whole lines
/// until they pass MAX_STDERR_BYTES; Err adds the last line, ended or not. A line with no
/// end is cut every MAX_STDERR_BYTES (the memory stays bounded), and its rest is never
/// taken for the ready line.
async fn read_ready_line<R: AsyncRead + Unpin>(stderr: &mut R) -> Result<String, String> {
    let mut before: Vec<u8> = Vec::new();
    let mut line: Vec<u8> = Vec::new();
    let mut cut = false; // `line` is the rest of a line that was cut
    loop {
        let byte = match stderr.read_u8().await {
            Ok(byte) => byte,
            Err(_) => {
                before.extend_from_slice(&line);
                return Err(String::from_utf8_lossy(&before).into_owned());
            }
        };
        line.push(byte);
        if byte != b'\n' && line.len() <= MAX_STDERR_BYTES {
            continue;
        }
        if !cut && line == READY_LINE {
            return Ok(String::from_utf8_lossy(&before).into_owned());
        }
        if before.len() < MAX_STDERR_BYTES {
            before.extend_from_slice(&line);
        }
        cut = byte != b'\n';
        line.clear();
    }
}

/// Exercise a REPL with a no-op, so the first exec of a user is not the REPL's first.
/// A new REPL has written its ready line (`take_or_spawn_repl`); the no-op loads the
/// modules that the REPL loads only on demand and starts its output paths. On a live
/// REPL (an L1 restore, a warm-up before a reclaim) it also maps its pages again.
///
/// Pre-advertisement paths only (warm-pool pre-warm and background L1
/// snapshot saves), hence the generous deadline: it runs before the VM is
/// available to callers.
async fn warm_exercise_repl(
    repl: &mut ReplState,
    language: Language,
) -> Result<(), Box<dyn std::error::Error>> {
    const PHASE: &str = "warm_pool";

    let (sentinel_id, sentinel_prefix) = generate_sentinel();

    // Language-appropriate no-op code. Python: load the modules that the REPL
    // loads only when user code needs them (a cold start does not wait for them),
    // with __import__, which binds no name: user code sees the same names on a
    // pooled VM as on a cold one. Python and JavaScript also print a value:
    // the first output of a REPL is slow (the stream and console paths start), and
    // the read below drops it. Measured on a pooled VM, first exec against the
    // second: HVF Python 4 / 1 ms, JavaScript 13 / 2 ms (TCG 55 / 3, 127 / 4) with
    // a no-op that printed nothing.
    let code: &[u8] = match language {
        Language::Python => {
            b"__import__('multiprocessing'), __import__('traceback'), __import__('typing'), print(0)"
        }
        Language::Javascript => b"console.log(0)",
        Language::Raw => b":",
    };

    // With the reseed flag: the REPL's reseed path runs once here, before any user code, so a
    // copy of this state runs it warm at its first exec. Measured on HVF, JavaScript (its
    // BoringSSL reseed): 16 ms at the first run after a restore, 1 ms warm.
    submit_code_to_repl(&mut repl.stdin, &sentinel_id, code, true).await?;

    log_info!(
        "[repl-readiness] phase={PHASE} language={} action=waiting timeout_ms={}",
        language.as_str(),
        WARM_UP_TIMEOUT.as_millis(),
    );

    // Wait for sentinel on stderr (this is where the interpreter startup time is spent)
    let mut stderr_line_buf = Vec::new();
    let mut stderr_discard = String::new();
    let mut buf = [0u8; 4096];
    let result = tokio::time::timeout(WARM_UP_TIMEOUT, async {
        loop {
            let n = repl.stderr.read(&mut buf).await?;
            if n == 0 {
                return Err("REPL stderr closed before sentinel".into());
            }
            if process_stderr_chunk(
                &buf[..n],
                &mut stderr_line_buf,
                &sentinel_prefix,
                &mut stderr_discard,
            )
            .is_some()
            {
                return Ok(());
            }
        }
    })
    .await;

    match repl.child.try_wait() {
        Ok(Some(status)) => {
            let exit_code = exit_code_from_status(status);
            crate::oom_guard::reconcile_retirement_exit(exit_code);
            return Err(format!("REPL exited during warm-up with code {exit_code}").into());
        }
        Ok(None) => {}
        Err(error) => return Err(format!("Failed to inspect warmed REPL: {error}").into()),
    }
    if crate::oom_guard::retirement_required() {
        return Err("VM retirement required during REPL warm-up".into());
    }

    match result {
        Ok(Ok(())) => {
            // Read the no-op's stdout up to its end marker, so no byte of it
            // reaches the next exec. 250 ms: TCG slack for the fallback.
            let mut end = StdoutEnd::new(&sentinel_id);
            drain_stdout(&mut repl.stdout, &mut end, &mut String::new(), &mut 0, 250).await;
            log_info!(
                "[repl-readiness] phase={PHASE} language={} action=ready",
                language.as_str()
            );
            Ok(())
        }
        Ok(Err(e)) => Err(e),
        Err(_) => {
            log_repl_timeout_diagnostics(repl.child.id(), false);
            Err(format!(
                "REPL {PHASE} readiness timed out after {}ms",
                WARM_UP_TIMEOUT.as_millis()
            )
            .into())
        }
    }
}

/// Execute code via persistent REPL with streaming output. `reseed`: the REPL reseeds
/// its own random generators from the kernel before the code runs.
pub(crate) async fn execute_code_streaming(
    language: Language,
    code: &str,
    timeout: u64,
    env_vars: &HashMap<String, String>,
    reseed: bool,
    writer: &ResponseWriter,
) -> Result<(), CmdError> {
    validate_code_params(code, timeout)?;
    validate_env_vars(env_vars)?;

    let start = Instant::now();
    let effective_timeout = effective_timeout_secs(timeout);
    let timeout_duration = Duration::from_secs(effective_timeout);
    // One time budget for a new REPL's start and the run, as when the code
    // waited in the pipe while the REPL started: the host's deadline holds
    let deadline = tokio::time::Instant::now() + timeout_duration;

    // Get or create persistent REPL for this language. A REPL that cannot
    // start (a retired VM included) is a reply, never a dropped connection:
    // the host sees no close of the port and would wait for its deadline.
    let spawn_start = Instant::now();
    let (mut repl, was_fresh_spawn) = take_or_spawn_repl(language, timeout_duration)
        .await
        .map_err(|e| CmdError::execution(format!("Failed to start REPL: {e}")))?;
    let spawn_ms = spawn_start.elapsed().as_millis() as u64;
    log_info!(
        "[timing] repl_acquired: {}ms (fresh={}, spawn={}ms)",
        crate::monotonic_ms(),
        was_fresh_spawn,
        spawn_ms
    );

    if let Err(e) = crate::oom_guard::ensure_vm_usable() {
        kill_repl(&mut repl).await;
        return Err(CmdError::execution(e.to_string()));
    }

    // Generate unique sentinel
    let (sentinel_id, sentinel_prefix) = generate_sentinel();

    // Prepend env var setup to user code
    let full_code = prepend_env_vars(language, code, env_vars);
    let code_bytes = full_code.as_bytes();

    // Write header + code to REPL stdin
    // A REPL started for this exec took its seeds from the kernel at its start: no reseed
    let write_result = submit_code_to_repl(
        &mut repl.stdin,
        &sentinel_id,
        code_bytes,
        reseed && !was_fresh_spawn,
    )
    .await;
    if let Err(e) = write_result {
        log_error!("REPL stdin write failed: {e}");
        kill_repl(&mut repl).await;
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
    let mut stderr_line_buf = Vec::new();
    let mut stdout_bytes = [0u8; OUTPUT_READ_BYTES];
    let mut stderr_bytes = [0u8; OUTPUT_READ_BYTES];
    let mut flush_timer = interval(Duration::from_millis(FLUSH_INTERVAL_MS));
    // Time of the last output sent (flush_due); None until the first.
    let mut last_flush: Option<Instant> = None;
    let mut stdout_done = false;
    let mut stderr_done = false;
    let mut sentinel_exit_code: Option<i32> = None;
    let mut stdout_end = StdoutEnd::new(&sentinel_id);

    // Output size cap counters (guest-enforced limits).
    // These count bytes, not UTF-8 decoded string length: stdout counts the
    // output left after the end marker is removed, stderr the raw pipe bytes.
    // When a stdout read crosses the limit, its whole chunk is dropped (the host
    // may get up to about 8 KiB less than the limit). The complete lines of the
    // stderr read that crosses it are still sent (up to about 8 KiB more). A
    // stream is capped once its total passes its cap (no flag: ExecOutput, after
    // the loop, uses the same rule). The drains after the loop (drain_stdout,
    // drain_after_sigterm) count with the same rules: the host stops a stream
    // past its cap and one read.
    let mut stdout_total: usize = 0;
    let mut stderr_total: usize = 0;

    let loop_result = tokio::time::timeout_at(deadline, async {
        loop {
            tokio::select! {
                _ = flush_timer.tick() => {
                    if !stdout_buffer.is_empty() || !stderr_buffer.is_empty() {
                        last_flush = Some(Instant::now());
                    }
                    flush_both(writer, &mut stdout_buffer, &mut stderr_buffer).await;
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
                            // The marker is found before the cap: a capped exec still ends at it.
                            // Past the cap the pipe is still read (the child never blocks), and dropped.
                            let output = stdout_end.feed(&stdout_bytes[..n]);
                            keep_under_stdout_cap(&output, &mut stdout_total, &mut stdout_buffer);
                            if stdout_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                last_flush = Some(Instant::now());
                                let _ = flush_output_buffer(writer, &mut stdout_buffer, stdout_frame).await;
                            } else if !stdout_buffer.is_empty() && flush_due(last_flush) {
                                // Both buffers, as the tick does: stderr output that waits
                                // for the tick goes out with this stdout
                                last_flush = Some(Instant::now());
                                flush_both(writer, &mut stdout_buffer, &mut stderr_buffer).await;
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
                            // Asymmetry with stdout: stderr must be parsed for the sentinel
                            // BEFORE the cap check, so process_stderr_chunk runs unconditionally.
                            // It writes into stderr_buffer even past the limit: the read that
                            // crosses it is flushed, the later ones are cleared below.
                            if let Some(code) = process_stderr_chunk(
                                &stderr_bytes[..n],
                                &mut stderr_line_buf,
                                &sentinel_prefix,
                                &mut stderr_buffer,
                            ) {
                                sentinel_exit_code = Some(code);
                            }
                            let was_capped = stderr_total > MAX_STDERR_BYTES;
                            stderr_total += n;
                            if stderr_total > MAX_STDERR_BYTES {
                                if !was_capped {
                                    // The crossing read: its lines and what came before go out
                                    let _ = flush_output_buffer(writer, &mut stderr_buffer, stderr_frame).await;
                                }
                                stderr_buffer.clear(); // discard parsed-but-over-limit data
                            } else if stderr_buffer.len() >= MAX_BUFFER_SIZE_BYTES {
                                last_flush = Some(Instant::now());
                                let _ = flush_output_buffer(writer, &mut stderr_buffer, stderr_frame).await;
                            } else if !stderr_buffer.is_empty() && flush_due(last_flush) {
                                // Both buffers, in the tick's order
                                last_flush = Some(Instant::now());
                                flush_both(writer, &mut stdout_buffer, &mut stderr_buffer).await;
                            }
                        }
                        Err(_) => stderr_done = true,
                    }
                }
            }

            if sentinel_exit_code.is_some() {
                break;
            }
            if stdout_done && stderr_done {
                break;
            }
        }
    })
    .await;

    let end = LoopEnd::of(loop_result.is_err(), sentinel_exit_code);
    let mut out = ExecOutput {
        stdout_buffer,
        stdout_end,
        stdout_total,
        stderr_buffer,
        stderr_line_buf,
        stderr_total,
    };
    if matches!(end, LoopEnd::Sentinel(_)) && !crate::oom_guard::retirement_required() {
        out.drain_trailing_stdout(&mut repl.stdout, 500).await;
    }
    out.flush_rest(writer, &end).await;

    let duration_ms = start.elapsed().as_millis() as u64;
    let process_ms = Some(process_start.elapsed().as_millis() as u64);
    let complete = |exit_code| GuestResponse::Complete {
        exit_code,
        execution_time_ms: duration_ms,
        spawn_ms: Some(spawn_ms),
        process_ms,
    };

    match end {
        LoopEnd::Sentinel(exit_code) => {
            // A terminal pressure decision can race with buffered sentinel
            // delivery. Do not pool the REPL or report a non-terminal result
            // once that decision is visible (see reconcile_retirement_exit).
            if crate::oom_guard::reconcile_retirement_exit(exit_code) == 137 {
                kill_repl(&mut repl).await;
                writer.send(&complete(137)).await?;
                return Ok(());
            }

            idle_repls().insert(language, repl);

            if let Some(message) = out.limit_message() {
                return Err(CmdError::output_limit(message));
            }

            writer.send(&complete(exit_code)).await?;
        }
        LoopEnd::ReplDied => {
            // Both pipes closed without a sentinel: the REPL has most likely
            // exited. One that still runs after KILL_WAIT (its code closed
            // fd 1 and 2) is killed.
            let status =
                match wait_child_bounded(&mut repl.child, "closing its output", KILL_WAIT).await {
                    Some(status) => Some(status),
                    None => {
                        graceful_terminate_process_group(&mut repl.child, 0).await;
                        repl.child.try_wait().ok().flatten()
                    }
                };
            let observed_exit_code = status.map_or(255, exit_code_from_status);
            let exit_code = crate::oom_guard::reconcile_retirement_exit(observed_exit_code);
            log_warn!(
                "REPL for {} died with exit_code={}",
                language.as_str(),
                exit_code
            );

            let exit_code = out.death_result(exit_code)?;
            writer.send(&complete(exit_code)).await?;
        }
        LoopEnd::Timeout => {
            log_repl_timeout_diagnostics(repl.child.id(), first_io_logged);
            log_warn!(
                "REPL for {} timed out after {}s, sending SIGTERM",
                language.as_str(),
                effective_timeout
            );
            graceful_terminate_process_group(&mut repl.child, TERM_GRACE_PERIOD_SECONDS).await;

            // Drain stdout/stderr after SIGTERM — the process may have printed
            // output (e.g. Python SIGTERM handler) and emitted a sentinel.
            let sigterm_exit_code = out
                .finish_after_sigterm(writer, &mut repl.stdout, &mut repl.stderr, &sentinel_prefix)
                .await;

            // If sentinel was found during post-SIGTERM drain, treat as graceful exit
            if let Some(exit_code) = sigterm_exit_code {
                // No output-limit error here: the REPL got SIGTERM and is not
                // pooled, so the sentinel code is the result.
                let exit_code = crate::oom_guard::reconcile_retirement_exit(exit_code);
                writer.send(&complete(exit_code)).await?;
                return Ok(());
            }

            if crate::oom_guard::retirement_required() {
                writer.send(&complete(137)).await?;
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

    #[rstest]
    #[case(false, "1_2 5 0\nprint")]
    #[case(true, "1_2 5 1\nprint")]
    #[tokio::test]
    async fn the_header_holds_the_id_the_code_length_and_the_reseed_flag(
        #[case] reseed: bool,
        #[case] expected: &str,
    ) {
        let mut stdin: Vec<u8> = Vec::new();
        submit_code_to_repl(&mut stdin, "1_2", b"print", reseed)
            .await
            .unwrap();
        assert_eq!(String::from_utf8(stdin).unwrap(), expected);
    }

    #[rstest]
    #[case("42 (awk) S 7 7 7 0 -1", Some(("awk", "S")))]
    #[case("42 (a) b) (c) D 7 7 7 0 -1", Some(("a) b) (c", "D")))]
    #[case("42 (a\n[WARN] b) S 7 7 7 0 -1", Some(("a?[WARN] b", "S")))] // no new log line
    #[case("42 (awk) S 7 9 9 0 -1", None)] // another group
    #[case("42 (awk) S", None)] // cut
    #[case("unavailable", None)]
    fn group_member_reads_name_and_state_of_a_member(
        #[case] stat: &str,
        #[case] expected: Option<(&str, &str)>,
    ) {
        assert_eq!(
            group_member(stat, 7),
            expected.map(|(comm, state)| (comm.to_string(), state))
        );
    }

    #[test]
    fn the_stdout_cap_keeps_up_to_its_last_byte() {
        let (mut total, mut buffer) = (MAX_STDOUT_BYTES - 3, String::new());
        keep_under_stdout_cap(b"abc", &mut total, &mut buffer);
        assert_eq!((total, buffer.as_str()), (MAX_STDOUT_BYTES, "abc")); // at the cap: kept
        keep_under_stdout_cap(b"d", &mut total, &mut buffer);
        keep_under_stdout_cap(b"e", &mut total, &mut buffer);
        assert_eq!((total, buffer.as_str()), (MAX_STDOUT_BYTES + 2, "abc"));
    }

    #[tokio::test]
    async fn a_drain_past_the_stdout_cap_keeps_only_the_reads_under_it() {
        // stdout that trails the sentinel: 60,000 B in the pipe (written whole before the
        // drain: no idle window ends it early), 10,000 B left under the cap
        let mut child = tokio::process::Command::new("sh")
            .args(["-c", "head -c 60000 /dev/zero | tr '\\0' x"])
            .stdout(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let mut stdout = child.stdout.take().unwrap();
        assert!(child.wait().await.unwrap().success());
        let mut end = StdoutEnd::new("1_2");
        let (mut total, mut buffer) = (MAX_STDOUT_BYTES - 10_000, String::new());

        drain_stdout(&mut stdout, &mut end, &mut buffer, &mut total, 5_000).await;

        assert_eq!(total, MAX_STDOUT_BYTES - 10_000 + 60_000); // every read counted, to the end
        assert_eq!(buffer, "x".repeat(8_192)); // the one read under the cap, then nothing
    }

    /// An ExecOutput with `stdout_total` and `stderr_total` counted before, nothing buffered
    /// (the stdout marker of exec "1_0").
    fn output_at(stdout_total: usize, stderr_total: usize) -> ExecOutput {
        ExecOutput {
            stdout_buffer: String::new(),
            stdout_end: StdoutEnd::new("1_0"),
            stdout_total,
            stderr_buffer: String::new(),
            stderr_line_buf: Vec::new(),
            stderr_total,
        }
    }

    /// A ResponseWriter, and the receiver of the lines it queues.
    fn writer() -> (ResponseWriter, tokio::sync::mpsc::Receiver<Vec<u8>>) {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        (ResponseWriter::new(tx, None), rx)
    }

    /// The (type, chunk) of each message queued in `rx`.
    fn sent(rx: &mut tokio::sync::mpsc::Receiver<Vec<u8>>) -> Vec<(String, String)> {
        let mut messages = Vec::new();
        while let Ok(line) = rx.try_recv() {
            let value: serde_json::Value = serde_json::from_slice(&line[..line.len() - 1]).unwrap();
            let chunk = value["chunk"].as_str().unwrap_or_default().to_string();
            messages.push((value["type"].as_str().unwrap().to_string(), chunk));
        }
        messages
    }

    /// A child that wrote `script`'s output and ended: its two pipes, at their end.
    async fn ended_pipes(
        script: &str,
    ) -> (tokio::process::ChildStdout, tokio::process::ChildStderr) {
        let mut child = tokio::process::Command::new("sh")
            .args(["-c", script])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let pipes = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        assert!(child.wait().await.unwrap().success());
        pipes
    }

    #[tokio::test]
    async fn the_timeout_drain_keeps_each_stream_under_its_cap() {
        // stdout 60,000 B with 10,000 B left under its cap; stderr 20 lines of 1,000 B with
        // 5,000 B left under its cap, then the sentinel. All of it is in the pipes before the
        // drain (under 64 KiB each), so each read is 8,192 B.
        let (mut stdout, mut stderr) = ended_pipes(
            "head -c 60000 /dev/zero | tr '\\0' x; i=0; while [ $i -lt 20 ]; do printf '%0999d\\n' 0 >&2; \
             i=$((i+1)); done; printf '__SENTINEL_1_0_7__\\n' >&2",
        )
        .await;
        let mut out = output_at(MAX_STDOUT_BYTES - 10_000, MAX_STDERR_BYTES - 5_000);

        let result = out
            .drain_after_sigterm(&mut stdout, &mut stderr, "__SENTINEL_1_0_")
            .await;

        assert_eq!(result, Some(7));
        assert_eq!(out.stdout_total, MAX_STDOUT_BYTES - 10_000 + 60_000);
        assert_eq!(out.stdout_buffer, "x".repeat(8_192)); // the one read under the cap
        assert_eq!(out.stderr_total, MAX_STDERR_BYTES - 5_000 + 20_000 + 19);
        // the whole lines of the read that crosses the cap (8 of its 8,192 B), then nothing
        assert_eq!(
            out.stderr_buffer,
            format!("{}\n", "0".repeat(999)).repeat(8)
        );
    }

    #[tokio::test]
    async fn the_timeout_drain_under_the_cap_keeps_the_partial_line() {
        let (mut stdout, mut stderr) =
            ended_pipes("printf 'late line\\n__SENTINEL_1_0_0__\\npartial' >&2").await;
        let mut out = output_at(0, 0);
        let result = out
            .drain_after_sigterm(&mut stdout, &mut stderr, "__SENTINEL_1_0_")
            .await;
        assert_eq!(
            (
                result,
                out.stderr_buffer.as_str(),
                out.stderr_line_buf.len()
            ),
            (Some(0), "late line\npartial", 0)
        );
    }

    #[test]
    fn a_stream_is_capped_past_its_cap() {
        // From the totals: the only cap state
        let past = output_at(MAX_STDOUT_BYTES + 1, MAX_STDERR_BYTES + 1);
        assert_eq!((past.stdout_capped(), past.stderr_capped()), (true, true));
        let at = output_at(MAX_STDOUT_BYTES, MAX_STDERR_BYTES);
        assert_eq!((at.stdout_capped(), at.stderr_capped()), (false, false));
    }

    #[tokio::test]
    async fn a_drain_that_crosses_the_cap_sends_what_came_before() {
        // 10,000 B left under the cap, "pre" not sent yet; 60,000 B trail the sentinel
        let (writer, mut rx) = writer();
        let (mut stdout, _stderr) = ended_pipes("head -c 60000 /dev/zero | tr '\\0' x").await;
        let mut out = output_at(MAX_STDOUT_BYTES - 10_000, 0);
        out.stdout_buffer.push_str("pre");

        out.drain_trailing_stdout(&mut stdout, 5_000).await;
        out.flush_rest(&writer, &LoopEnd::Sentinel(0)).await;

        let before = format!("pre{}", "x".repeat(8_192)); // the one read under the cap
        assert_eq!(sent(&mut rx), [("stdout".to_string(), before)]);
        assert!(out.stdout_capped());
    }

    #[tokio::test]
    async fn a_character_split_at_the_deadline_comes_out_whole() {
        // A read ended inside "é" of "café": the timeout flush leaves the partial line to
        // the drain, which reads the rest of the character
        let (writer, mut rx) = writer();
        let (mut stdout, mut stderr) =
            ended_pipes("printf '\\251\\n__SENTINEL_1_0_0__\\n' >&2").await;
        let mut out = output_at(0, 0);
        out.stderr_line_buf.extend_from_slice(b"caf\xc3");

        out.flush_rest(&writer, &LoopEnd::Timeout).await;
        assert_eq!(sent(&mut rx), Vec::<(String, String)>::new());
        let result = out
            .finish_after_sigterm(&writer, &mut stdout, &mut stderr, "__SENTINEL_1_0_")
            .await;

        assert_eq!(
            (result, sent(&mut rx)),
            (Some(0), vec![("stderr".to_string(), "café\n".to_string())])
        );
    }

    #[tokio::test]
    async fn a_sentinel_line_split_at_the_deadline_still_ends_the_drain() {
        // The REPL wrote its sentinel; the deadline cut its line: no piece of it goes out
        let (writer, mut rx) = writer();
        let (mut stdout, mut stderr) = ended_pipes("printf '0__\\n' >&2").await;
        let mut out = output_at(0, 0);
        out.stderr_line_buf.extend_from_slice(b"__SENTINEL_1_0_");

        out.flush_rest(&writer, &LoopEnd::Timeout).await;
        let result = out
            .finish_after_sigterm(&writer, &mut stdout, &mut stderr, "__SENTINEL_1_0_")
            .await;

        assert_eq!((result, sent(&mut rx)), (Some(0), vec![]));
    }

    #[rstest]
    #[case::at_the_cap(MAX_STDERR_BYTES, "kept\npart")]
    #[case::past_the_cap(MAX_STDERR_BYTES + 1, "kept\n")]
    #[tokio::test]
    async fn an_ended_exec_sends_stderr_and_its_partial_line_up_to_the_cap(
        #[case] stderr_total: usize,
        #[case] expected: &str,
    ) {
        // The buffer holds what may go out (also past the cap: the crossing read's lines);
        // the partial line goes with it only while stderr is at its cap or under
        let (writer, mut rx) = writer();
        let mut out = output_at(0, stderr_total);
        out.stderr_buffer.push_str("kept\n");
        out.stderr_line_buf.extend_from_slice(b"part");
        out.flush_rest(&writer, &LoopEnd::Sentinel(0)).await;
        assert_eq!(
            sent(&mut rx),
            [("stderr".to_string(), expected.to_string())]
        );
    }

    #[tokio::test]
    async fn the_timeout_drain_keeps_the_partial_line_up_to_the_cap() {
        // 26 B read: the sentinel line (19 B) and "partial" (7 B) after it
        let text = "printf '__SENTINEL_1_0_0__\\npartial' >&2";
        for (left, kept) in [(26, "partial"), (25, "")] {
            let (mut stdout, mut stderr) = ended_pipes(text).await;
            let mut out = output_at(0, MAX_STDERR_BYTES - left);
            out.drain_after_sigterm(&mut stdout, &mut stderr, "__SENTINEL_1_0_")
                .await;
            assert_eq!(out.stderr_buffer, kept, "{left} B left under the cap");
        }
    }

    #[tokio::test]
    async fn the_timeout_drain_counts_its_held_tail() {
        // 4 B left under the cap: "ab" is kept, the held "__STDOUT_EN" (no marker after it) is not
        let (mut stdout, mut stderr) = ended_pipes("printf 'ab__STDOUT_EN'").await;
        let mut out = output_at(MAX_STDOUT_BYTES - 4, 0);
        out.drain_after_sigterm(&mut stdout, &mut stderr, "__SENTINEL_1_0_")
            .await;
        assert_eq!(
            (out.stdout_buffer.as_str(), out.stdout_capped()),
            ("ab", true)
        );
    }

    #[tokio::test]
    async fn a_timeout_flush_leaves_a_split_marker_to_the_drain() {
        // The deadline cut the marker "__STDOUT_END_1_0__\n": its start is held
        let (writer, mut rx) = writer();
        let mut out = output_at(0, 0);
        let output = out.stdout_end.feed(b"abc__STDOUT_END_1_");
        out.stdout_buffer
            .push_str(&String::from_utf8_lossy(&output));

        out.flush_rest(&writer, &LoopEnd::Timeout).await;

        assert_eq!(sent(&mut rx), [("stdout".to_string(), "abc".to_string())]);
        assert!(out.stdout_end.feed(b"0__\n").is_empty() && out.stdout_end.seen); // the drain's read ends it
    }

    #[tokio::test]
    async fn an_ended_exec_counts_its_held_tail() {
        // 4 B left under the cap: "ab" goes out, the held "__STDOUT_EN" (11 B: no marker after it) does not
        let (writer, mut rx) = writer();
        let mut out = output_at(MAX_STDOUT_BYTES - 4, 0);
        let output = out.stdout_end.feed(b"ab__STDOUT_EN");
        keep_under_stdout_cap(&output, &mut out.stdout_total, &mut out.stdout_buffer);

        out.flush_rest(&writer, &LoopEnd::ReplDied).await;

        assert_eq!(sent(&mut rx), [("stdout".to_string(), "ab".to_string())]);
        assert_eq!(
            out.limit_message().as_deref(),
            Some(
                format!(
                    "stdout {} bytes exceeds {} limit",
                    MAX_STDOUT_BYTES + 9,
                    MAX_STDOUT_BYTES
                )
                .as_str()
            )
        );
    }

    #[test]
    fn the_limit_message_names_each_capped_stream() {
        assert_eq!(
            output_at(MAX_STDOUT_BYTES, MAX_STDERR_BYTES).limit_message(),
            None
        );
        let out = output_at(MAX_STDOUT_BYTES + 5, MAX_STDERR_BYTES + 7);
        assert_eq!(
            out.limit_message().unwrap(),
            format!(
                "stdout {} bytes exceeds {} limit, stderr {} bytes exceeds {} limit",
                MAX_STDOUT_BYTES + 5,
                MAX_STDOUT_BYTES,
                MAX_STDERR_BYTES + 7,
                MAX_STDERR_BYTES
            )
        );
    }

    #[tokio::test]
    async fn after_a_timeout_a_drain_that_crosses_the_caps_sends_what_it_kept() {
        // 10,000 B left under the stdout cap, 5,000 under the stderr cap; the REPL handled
        // SIGTERM: its sentinel ends the drain
        let (writer, mut rx) = writer();
        let (mut stdout, mut stderr) = ended_pipes(
            "head -c 60000 /dev/zero | tr '\\0' x; i=0; while [ $i -lt 20 ]; do printf '%0999d\\n' 0 >&2; \
             i=$((i+1)); done; printf '__SENTINEL_1_0_7__\\n' >&2",
        )
        .await;
        let mut out = output_at(MAX_STDOUT_BYTES - 10_000, MAX_STDERR_BYTES - 5_000);

        let result = out
            .finish_after_sigterm(&writer, &mut stdout, &mut stderr, "__SENTINEL_1_0_")
            .await;

        assert_eq!(result, Some(7));
        assert_eq!(
            sent(&mut rx),
            [
                ("stdout".to_string(), "x".repeat(8_192)),
                (
                    "stderr".to_string(),
                    format!("{}\n", "0".repeat(999)).repeat(8)
                ),
            ]
        );
    }

    #[rstest]
    #[case::under_the_caps(0, b"Progress [__", 0, b"Processing...", vec![("stdout", "__"), ("stderr", "Processing...")])]
    #[case::split_characters(0, b"caf\xc3", 0, b"caf\xc3", vec![("stderr", "caf")])]
    #[case::at_the_caps(MAX_STDOUT_BYTES - 2, b"__", MAX_STDERR_BYTES, b"Processing...", vec![("stdout", "__"), ("stderr", "Processing...")])]
    #[case::past_the_caps(MAX_STDOUT_BYTES - 1, b"__", MAX_STDERR_BYTES + 1, b"Processing...", vec![])]
    #[tokio::test]
    async fn after_a_timeout_with_no_sentinel_the_text_held_at_the_deadline_goes_out(
        #[case] stdout_total: usize,
        #[case] stdout_read: &[u8],
        #[case] stderr_total: usize,
        #[case] stderr_line: &[u8],
        #[case] expected: Vec<(&str, &str)>,
    ) {
        // No SIGTERM handler, so no sentinel: the text held at the deadline goes out (the
        // stdout tail that may start the marker, the partial stderr line), the drain's reads
        // ("late", "more") do not. A character whose rest came after the deadline does not.
        let (writer, mut rx) = writer();
        let (mut stdout, mut stderr) = ended_pipes("printf late; printf 'more\\n' >&2").await;
        let mut out = output_at(stdout_total, stderr_total);
        out.stdout_end.feed(stdout_read); // the loop sent what this returns
        out.stderr_line_buf.extend_from_slice(stderr_line);
        let result = out
            .finish_after_sigterm(&writer, &mut stdout, &mut stderr, "__SENTINEL_1_0_")
            .await;
        let expected: Vec<(String, String)> = expected
            .into_iter()
            .map(|(t, c)| (t.to_string(), c.to_string()))
            .collect();
        assert_eq!((result, sent(&mut rx)), (None, expected));
    }

    #[tokio::test]
    async fn after_a_timeout_with_no_sentinel_nothing_goes_out() {
        let (writer, mut rx) = writer();
        let (mut stdout, mut stderr) = ended_pipes("printf out; printf 'late line\\n' >&2").await;
        let mut out = output_at(0, 0);
        let result = out
            .finish_after_sigterm(&writer, &mut stdout, &mut stderr, "__SENTINEL_1_0_")
            .await;
        assert_eq!((result, sent(&mut rx)), (None, vec![]));
    }

    #[rstest]
    #[case::past_the_cap(MAX_STDERR_BYTES + 1, vec![("stdout", "out")])]
    #[case::at_the_cap(MAX_STDERR_BYTES, vec![("stdout", "out"), ("stderr", "late line\n")])]
    #[tokio::test]
    async fn after_a_timeout_the_drain_keeps_stderr_up_to_the_read_that_crosses_its_cap(
        #[case] stderr_total: usize,
        #[case] expected: Vec<(&str, &str)>,
    ) {
        // At the cap the drain's first read crosses it: its lines are kept, then nothing
        let (writer, mut rx) = writer();
        let (mut stdout, mut stderr) =
            ended_pipes("printf out; printf 'late line\\n__SENTINEL_1_0_0__\\n' >&2").await;
        let mut out = output_at(0, stderr_total);
        let result = out
            .finish_after_sigterm(&writer, &mut stdout, &mut stderr, "__SENTINEL_1_0_")
            .await;
        let expected: Vec<(String, String)> = expected
            .into_iter()
            .map(|(t, c)| (t.to_string(), c.to_string()))
            .collect();
        assert_eq!((result, sent(&mut rx)), (Some(0), expected));
    }

    #[tokio::test]
    async fn text_under_the_cap_goes_out_after_the_stream_crosses_it() {
        // "abc" waited for a flush (or a deadline cut one) when a later read crossed the cap
        let (writer, mut rx) = writer();
        let mut out = output_at(MAX_STDOUT_BYTES + 1, 0);
        out.stdout_buffer.push_str("abc");
        out.flush_rest(&writer, &LoopEnd::Sentinel(0)).await;
        assert_eq!(sent(&mut rx), [("stdout".to_string(), "abc".to_string())]);
        assert_eq!(
            out.limit_message().unwrap(),
            format!(
                "stdout {} bytes exceeds {} limit",
                MAX_STDOUT_BYTES + 1,
                MAX_STDOUT_BYTES
            )
        );
    }

    #[tokio::test]
    async fn a_flush_cut_by_a_cancel_keeps_its_text() {
        // A full queue (one slot, taken) and the exec's deadline at 50 ms
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let writer = ResponseWriter::new(tx, None);
        writer
            .send(&stdout_frame("fill".to_string()))
            .await
            .unwrap();
        let mut buffer = "kept".to_string();

        let flush = flush_output_buffer(&writer, &mut buffer, stdout_frame);
        let cut = tokio::time::timeout(Duration::from_millis(50), flush)
            .await
            .is_err();

        assert!(cut);
        assert_eq!(buffer, "kept"); // not taken: the flush after the loop sends it
        assert_eq!(sent(&mut rx), [("stdout".to_string(), "fill".to_string())]);
        flush_output_buffer(&writer, &mut buffer, stdout_frame)
            .await
            .unwrap();
        assert_eq!(sent(&mut rx), [("stdout".to_string(), "kept".to_string())]);
    }

    #[rstest]
    #[case::past_the_cap(
        true,
        3,
        Err("stdout 1000001 bytes exceeds 1000000 limit; the REPL then exited with code 3")
    )]
    #[case::retire_137(true, 137, Ok(137))] // the VM must retire: the host sees 137
    #[case::under_the_cap(false, 3, Ok(3))]
    fn the_result_of_a_repl_that_died(
        #[case] past_cap: bool,
        #[case] exit_code: i32,
        #[case] expected: Result<i32, &str>,
    ) {
        let out = output_at(MAX_STDOUT_BYTES + usize::from(past_cap), 0);
        let result = out.death_result(exit_code).map_err(|error| match error {
            CmdError::Reply {
                message,
                error_type,
            } => {
                assert_eq!(error_type, "output_limit_error");
                message
            }
            other => panic!("{other:?}"),
        });
        assert_eq!(result, expected.map_err(String::from));
    }

    #[test]
    fn a_sentinel_read_before_the_deadline_ends_the_exec() {
        assert_eq!(LoopEnd::of(true, Some(0)), LoopEnd::Sentinel(0)); // the deadline cut a flush after it
        assert_eq!(LoopEnd::of(false, Some(3)), LoopEnd::Sentinel(3));
        assert_eq!(LoopEnd::of(false, None), LoopEnd::ReplDied);
        assert_eq!(LoopEnd::of(true, None), LoopEnd::Timeout);
    }

    #[tokio::test]
    async fn group_members_shows_where_a_member_waits() {
        // A leader that waits for its member, as a REPL waits for the command of an exec
        let mut leader = tokio::process::Command::new("sh")
            .args(["-c", "sleep 30 & wait"])
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let pid = leader.id().unwrap();
        let mut members = String::new();
        for _ in 0..100 {
            members = group_members_diagnostic(pid);
            // Its state is R for a moment after the exec: wait for the sleep itself
            if members.contains(" sleep S ") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        // SAFETY: kill(2) on the group this test started
        unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
        let _ = leader.wait().await;

        assert!(members.contains(" sleep S wchan="), "{members}");
        assert!(!members.contains(&format!("{pid} sh ")), "{members}"); // the leader has its own fields
    }

    #[tokio::test]
    async fn ready_line_returns_what_came_before_and_leaves_the_rest_in_the_pipe() {
        let mut stderr: &[u8] = b"a warning\n__REPL_READY__\nnext exec";
        assert_eq!(
            read_ready_line(&mut stderr).await,
            Ok("a warning\n".to_string())
        );
        assert_eq!(stderr, b"next exec");
    }

    #[tokio::test]
    async fn ready_line_alone_returns_nothing_before() {
        let mut stderr: &[u8] = b"__REPL_READY__\n";
        assert_eq!(read_ready_line(&mut stderr).await, Ok(String::new()));
    }

    #[tokio::test]
    async fn stderr_closed_before_the_ready_line_is_a_failed_start() {
        let mut stderr: &[u8] = b"Error: prctl(PR_SET_DUMPABLE, 0) failed\n    at js_repl.mjs:21";
        assert_eq!(
            read_ready_line(&mut stderr).await,
            Err("Error: prctl(PR_SET_DUMPABLE, 0) failed\n    at js_repl.mjs:21".to_string())
        );
    }

    #[tokio::test]
    async fn a_line_with_no_end_is_cut_and_its_rest_is_not_the_ready_line() {
        let mut input = vec![b'x'; MAX_STDERR_BYTES];
        input.extend_from_slice(b"y__REPL_READY__\n__REPL_READY__\n");
        let mut stderr: &[u8] = &input;
        let before = read_ready_line(&mut stderr)
            .await
            .expect("the ready line on its own");
        assert_eq!(before.len(), MAX_STDERR_BYTES + 1); // the cut start; then full: the rest is dropped
        assert!(stderr.is_empty());
    }

    #[tokio::test]
    async fn the_ready_text_inside_a_line_is_not_the_ready_line() {
        let mut stderr: &[u8] = b"x__REPL_READY__\n";
        assert_eq!(
            read_ready_line(&mut stderr).await,
            Err("x__REPL_READY__\n".to_string())
        );
    }

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

    #[rstest]
    #[case("-1", 255)]
    #[case("256", 0)]
    #[case("-256", 0)]
    #[case("2147483647", 255)]
    fn sentinel_out_of_range_exit_code_is_a_process_status(
        #[case] code: &str,
        #[case] expected: i32,
    ) {
        // Like a process exit status: -1 stays the host's timeout code only.
        let mut buf = String::new();
        let line = format!("__SENTINEL_1_0_{code}__");
        let result = parse_stderr_line_for_sentinel(&line, "__SENTINEL_1_0_", &mut buf);
        assert_eq!(result, Some(expected));
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
        // "42_extra" is not a valid i32, falls back to 255 (unknown error)
        assert_eq!(result, Some(255));
    }

    #[test]
    fn sentinel_non_numeric_exit_code() {
        let mut buf = String::new();
        let result =
            parse_stderr_line_for_sentinel("__SENTINEL_1_0_abc__", "__SENTINEL_1_0_", &mut buf);
        assert_eq!(result, Some(255));
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
        // Empty string between prefix and __ → parse fails → 255
        assert_eq!(result, Some(255));
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
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let chunk = b"__SENTINEL_1_0_0__\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, Some(0));
        assert!(line_buf.is_empty());
        assert!(stderr_buf.is_empty());
    }

    #[test]
    fn chunk_single_plain_line() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let chunk = b"hello world\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert!(line_buf.is_empty(), "complete line should be consumed");
        assert_eq!(stderr_buf, "hello world\n");
    }

    #[test]
    fn chunk_multiple_lines_sentinel_on_last() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let chunk = b"line1\nline2\n__SENTINEL_1_0_5__\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, Some(5));
        assert_eq!(stderr_buf, "line1\nline2\n");
    }

    #[test]
    fn chunk_multiple_lines_sentinel_in_middle() {
        let mut line_buf = Vec::new();
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
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let chunk = b"partial data no newline";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(line_buf, b"partial data no newline");
        assert!(stderr_buf.is_empty(), "no complete lines yet");
    }

    #[test]
    fn chunk_split_across_two_deliveries() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let prefix = "__SENTINEL_1_0_";

        // First chunk: partial line
        let r1 = process_stderr_chunk(b"__SENTINEL_1", &mut line_buf, prefix, &mut stderr_buf);
        assert_eq!(r1, None);
        assert_eq!(line_buf, b"__SENTINEL_1");

        // Second chunk: completes the line
        let r2 = process_stderr_chunk(b"_0_0__\n", &mut line_buf, prefix, &mut stderr_buf);
        assert_eq!(r2, Some(0));
        assert!(line_buf.is_empty());
    }

    #[test]
    fn chunk_sentinel_split_across_three_deliveries() {
        let mut line_buf = Vec::new();
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
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let result = process_stderr_chunk(b"", &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert!(line_buf.is_empty());
        assert!(stderr_buf.is_empty());
    }

    #[test]
    fn chunk_only_newlines() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let result =
            process_stderr_chunk(b"\n\n\n", &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "\n\n\n");
    }

    #[test]
    fn chunk_invalid_utf8_lossy() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        // 0xFF is invalid UTF-8, should be replaced with U+FFFD
        let chunk: &[u8] = &[0xFF, b'\n'];
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "\u{FFFD}\n");
    }

    #[test]
    fn chunk_very_long_line() {
        let mut line_buf = Vec::new();
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
        let mut line_buf = Vec::new();
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
        let mut line_buf = b"leftover".to_vec();
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
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let chunk = b"complete\npartial";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "complete\n");
        assert_eq!(line_buf, b"partial");
    }

    // -------------------------------------------------------------------------
    // process_stderr_chunk — out-of-bound / adversarial cases
    // -------------------------------------------------------------------------

    #[test]
    fn chunk_fake_sentinel_wrong_id() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let chunk = b"__SENTINEL_999_999_0__\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "__SENTINEL_999_999_0__\n");
    }

    #[test]
    fn chunk_sentinel_like_pattern_incomplete() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        // Looks like sentinel but missing trailing __
        let chunk = b"__SENTINEL_1_0_42\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, None);
        assert_eq!(stderr_buf, "__SENTINEL_1_0_42\n");
    }

    #[test]
    fn chunk_multiple_sentinels_last_wins() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let chunk = b"__SENTINEL_1_0_1__\n__SENTINEL_1_0_2__\n";
        let result = process_stderr_chunk(chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        // Last exit code wins (loop continues after first sentinel)
        assert_eq!(result, Some(2));
    }

    #[test]
    fn chunk_sentinel_with_binary_noise_before() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let mut chunk: Vec<u8> = vec![0x00, 0x01, 0x02];
        chunk.extend_from_slice(b"__SENTINEL_1_0_7__\n");
        let result =
            process_stderr_chunk(&chunk, &mut line_buf, "__SENTINEL_1_0_", &mut stderr_buf);
        assert_eq!(result, Some(7));
    }

    #[rstest]
    #[case(i32::MAX, 255)]
    #[case(i32::MIN, 0)]
    fn chunk_sentinel_boundary_exit_codes(#[case] code: i32, #[case] kept: i32) {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let prefix = "__SENTINEL_1_0_";
        let line = format!("{prefix}{code}__\n");
        let result = process_stderr_chunk(line.as_bytes(), &mut line_buf, prefix, &mut stderr_buf);
        // Kept like a process exit status: i32::MAX is 255, i32::MIN is 0.
        assert_eq!(result, Some(kept));
    }

    #[test]
    fn newline_free_flood_keeps_line_buffer_bounded() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let prefix = "__SENTINEL_1_0_";
        let chunk = vec![b'x'; 256 * 1024];
        // Many chunks, never a newline: the tail keeps its last 100000 bytes (the stderr
        // limit) and the length of a sentinel prefix, and grows no more.
        for _ in 0..40 {
            assert_eq!(
                process_stderr_chunk(&chunk, &mut line_buf, prefix, &mut stderr_buf),
                None
            );
            assert_eq!(line_buf.len(), 100_000 + 15);
        }
        assert_eq!(stderr_buf, "");
    }

    #[test]
    fn sentinel_survives_after_newline_free_flood() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let prefix = "__SENTINEL_1_0_";
        let flood = vec![b'x'; 2 * 1024 * 1024];
        process_stderr_chunk(&flood, &mut line_buf, prefix, &mut stderr_buf);
        // A terminated sentinel line delivered after the flood is still parsed.
        let result = process_stderr_chunk(
            b"\n__SENTINEL_1_0_5__\n",
            &mut line_buf,
            prefix,
            &mut stderr_buf,
        );
        assert_eq!(result, Some(5));
    }

    // -------------------------------------------------------------------------
    // flush_due — the first output after a quiet time goes out soon
    // -------------------------------------------------------------------------

    #[test]
    fn flush_is_due_before_the_first_flush_and_after_a_quiet_time() {
        let interval = Duration::from_millis(FLUSH_INTERVAL_MS);
        assert!(flush_due(None));
        assert!(
            !flush_due(Some(Instant::now())),
            "a flush just now: wait for the tick"
        );
        let quiet = Instant::now().checked_sub(interval).unwrap();
        assert!(flush_due(Some(quiet)));
    }

    // -------------------------------------------------------------------------
    // StdoutEnd — the per-exec stdout end marker
    // -------------------------------------------------------------------------

    const MARKER: &[u8] = b"__STDOUT_END_7_1__\n";

    fn fed(chunks: &[&[u8]]) -> (Vec<u8>, bool, Vec<u8>) {
        let mut end = StdoutEnd::new("7_1");
        let mut output = Vec::new();
        for chunk in chunks {
            output.extend(end.feed(chunk));
        }
        let seen = end.seen;
        (output, seen, end.finish())
    }

    #[test]
    fn marker_in_one_chunk_is_removed() {
        let chunk = [b"hello\n".as_slice(), MARKER].concat();
        assert_eq!(fed(&[&chunk]), (b"hello\n".to_vec(), true, Vec::new()));
    }

    #[test]
    fn marker_split_byte_by_byte_is_found() {
        let stream = [b"no newline".as_slice(), MARKER].concat();
        let bytes: Vec<&[u8]> = stream.chunks(1).collect();
        assert_eq!(fed(&bytes), (b"no newline".to_vec(), true, Vec::new()));
    }

    #[test]
    fn output_after_the_marker_stays() {
        let chunk = [b"a".as_slice(), MARKER, b"child\n"].concat();
        assert_eq!(
            fed(&[&chunk, b"more"]),
            (b"achild\nmore".to_vec(), true, Vec::new())
        );
    }

    #[test]
    fn another_exec_marker_is_output() {
        let (output, seen, _) = fed(&[b"__STDOUT_END_8_1__\n"]);
        assert_eq!((output, seen), (b"__STDOUT_END_8_1__\n".to_vec(), false));
    }

    #[test]
    fn only_a_marker_prefix_is_held_back() {
        let mut end = StdoutEnd::new("7_1");
        assert_eq!(end.feed(b"line\n__STDOUT_E"), b"line\n".to_vec());
        assert_eq!(end.feed(b"xtra"), b"__STDOUT_Extra".to_vec());
        assert_eq!(end.feed(b"__STD"), Vec::<u8>::new());
        // No marker came (fd 1 closed by user code): the tail is still output.
        assert_eq!(end.finish(), b"__STD".to_vec());
    }

    // A UTF-8 character that a pipe read split (here "中", 3 bytes, in two reads) comes out
    // whole, not as U+FFFD
    #[test]
    fn stderr_character_split_across_reads_stays_whole() {
        let mut line_buf = Vec::new();
        let mut stderr_buf = String::new();
        let bytes = "中".as_bytes();
        assert_eq!(
            process_stderr_chunk(
                &bytes[..2],
                &mut line_buf,
                "__SENTINEL_1_0_",
                &mut stderr_buf
            ),
            None
        );
        assert_eq!(
            process_stderr_chunk(
                &[bytes[2], b'\n'],
                &mut line_buf,
                "__SENTINEL_1_0_",
                &mut stderr_buf
            ),
            None
        );
        assert_eq!(stderr_buf, "中\n");
    }

    #[test]
    fn stdout_character_split_across_reads_stays_whole() {
        let mut end = StdoutEnd::new("1_0");
        let bytes = "a中".as_bytes();
        let first = end.feed(&bytes[..2]); // "a" and the first byte of "中"
        assert_eq!(first, b"a");
        let mut second = end.feed(&bytes[2..]);
        second.extend(end.finish());
        assert_eq!(String::from_utf8(second).unwrap(), "中");
    }

    #[rstest]
    #[case(b"abc".as_slice(), 0)]
    #[case(&[0xE4, 0xB8], 2)] // 2 of the 3 bytes of a character
    #[case(&[0xE4], 1)]
    #[case("中".as_bytes(), 0)]
    #[case(&[0xF0, 0x9F, 0x98], 3)] // 3 of the 4 bytes of an emoji
    #[case(&[0xFF], 0)] // no character starts with it: nothing to wait for
    fn incomplete_utf8_tail_counts_the_missing_character(
        #[case] bytes: &[u8],
        #[case] tail: usize,
    ) {
        assert_eq!(incomplete_utf8_tail(bytes), tail);
    }

    // 100,003 bytes of a control character with no newline (a stderr line before the
    // sentinel can hold that much): frames of at most 64 KB of text, each line below the
    // host's 512 KB limit, the text whole and in order
    #[tokio::test]
    async fn a_large_buffer_goes_out_in_frames_below_the_host_limit() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(16);
        let writer = ResponseWriter::new(tx, Some("op".into()));
        // "中" (3 B) across byte 65,536: the frame edge walks back to its start
        let text = format!("{}中{}", "\u{1}".repeat(65_535), "\u{1}".repeat(34_465));
        let mut buffer = text.clone();
        flush_output_buffer(&writer, &mut buffer, stderr_frame)
            .await
            .unwrap();
        drop(writer);
        assert!(buffer.is_empty());
        let mut sent_text = String::new();
        let mut frame_sizes = Vec::new();
        while let Some(line) = rx.recv().await {
            assert!(line.len() < 512 * 1024, "a frame of {} bytes", line.len());
            let value: serde_json::Value = serde_json::from_slice(&line[..line.len() - 1]).unwrap();
            assert_eq!(value["type"], "stderr");
            let chunk = value["chunk"].as_str().unwrap();
            assert!(chunk.len() <= MAX_BUFFER_SIZE_BYTES);
            sent_text.push_str(chunk);
            frame_sizes.push(chunk.len());
        }
        assert_eq!(frame_sizes, [65_535, 34_468]); // the edge before "中", not inside it
        assert_eq!(sent_text, text);
    }
}

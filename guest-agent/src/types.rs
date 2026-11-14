//! Wire protocol type definitions for the guest agent.
//!
//! All structs for inbound commands (host → guest) and outbound messages
//! (guest → host). Internal state types live in their owning modules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Inbound commands (host → guest, deserialized from JSON)
// ============================================================================

/// A host request, tagged by `action`. The host models are in
/// src/exec_sandbox/guest_agent_protocol.py. connection.rs reads the top-level
/// `op_id` and `host_time_ns` from the raw JSON before this parse.
#[derive(Debug, Deserialize)]
#[serde(tag = "action")]
pub(crate) enum GuestCommand {
    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "install_packages")]
    InstallPackages {
        language: Language,
        packages: Vec<String>,
        /// Seconds for the package manager process; 0 means MAX_TIMEOUT_SECONDS.
        timeout: u64,
    },

    #[serde(rename = "exec")]
    ExecuteCode {
        language: Language,
        code: String,
        /// Seconds, at most MAX_TIMEOUT_SECONDS; 0 means MAX_TIMEOUT_SECONDS.
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

    /// One chunk of a write_file stream: base64 of zstd-compressed bytes.
    #[serde(rename = "file_chunk")]
    FileChunk { op_id: String, data: String },

    /// End of a write_file stream: the guest commits the file and replies
    /// with FileWriteAck.
    #[serde(rename = "file_end")]
    FileEnd { op_id: String },

    #[serde(rename = "list_files")]
    ListFiles {
        #[serde(default)]
        path: String,
    },

    #[serde(rename = "warm_repl")]
    WarmRepl { language: Language },
}

// ============================================================================
// Outbound messages (guest → host, serialized to JSON)
// ============================================================================

/// A guest message, tagged by `type`. ResponseWriter adds the `op_id` of the
/// request to each message.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum GuestResponse {
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
    Error { message: String, error_type: String },

    #[serde(rename = "file_write_ack")]
    FileWriteAck { path: String, bytes_written: usize },

    #[serde(rename = "file_chunk")]
    FileChunk { data: String },

    #[serde(rename = "file_read_complete")]
    FileReadComplete { path: String, size: u64 },

    #[serde(rename = "file_list")]
    FileList { entries: Vec<FileEntry> },

    #[serde(rename = "warm_repl_ack")]
    WarmReplAck {
        language: Language,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
}

#[derive(Debug, Serialize)]
pub(crate) struct FileEntry {
    pub(crate) name: String,
    pub(crate) is_dir: bool,
    pub(crate) size: u64,
}

// ============================================================================
// Language dispatch
// ============================================================================

/// Supported execution languages. Their wire names are those of the host's
/// `Language` enum; a request with another name does not parse (request_error).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Language {
    Python,
    Javascript,
    Raw,
}

impl Language {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Javascript => "javascript",
            Self::Raw => "raw",
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Language
    // -------------------------------------------------------------------------

    #[test]
    fn test_language_wire_name_is_its_log_name() {
        for lang in [Language::Python, Language::Javascript, Language::Raw] {
            let wire = format!("\"{}\"", lang.as_str());
            assert_eq!(serde_json::to_string(&lang).unwrap(), wire);
            assert_eq!(serde_json::from_str::<Language>(&wire).unwrap(), lang);
        }
    }

    #[test]
    fn test_language_other_names_do_not_parse() {
        for name in [r#""PYTHON""#, r#""""#, r#""cobol""#] {
            assert!(serde_json::from_str::<Language>(name).is_err(), "{name}");
        }
    }

    // -------------------------------------------------------------------------
    // GuestCommand deserialization
    // -------------------------------------------------------------------------

    #[test]
    fn test_deserialize_write_file() {
        let json =
            r#"{"action":"write_file","op_id":"abc123","path":"test.txt","make_executable":false}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::WriteFile {
                op_id,
                path,
                make_executable,
            } => {
                assert_eq!(op_id, "abc123");
                assert_eq!(path, "test.txt");
                assert!(!make_executable);
            }
            _ => panic!("Expected WriteFile"),
        }
    }

    #[test]
    fn test_deserialize_read_file() {
        let json = r#"{"action":"read_file","op_id":"def456","path":"output.csv"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::ReadFile { op_id, path } => {
                assert_eq!(op_id, "def456");
                assert_eq!(path, "output.csv");
            }
            _ => panic!("Expected ReadFile"),
        }
    }

    #[test]
    fn test_deserialize_list_files() {
        let json = r#"{"action":"list_files","path":""}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::ListFiles { path } => assert_eq!(path, ""),
            _ => panic!("Expected ListFiles"),
        }
    }

    #[test]
    fn test_deserialize_list_files_default_path() {
        let json = r#"{"action":"list_files"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::ListFiles { path } => assert_eq!(path, ""),
            _ => panic!("Expected ListFiles"),
        }
    }

    #[test]
    fn test_deserialize_write_file_executable() {
        let json =
            r#"{"action":"write_file","op_id":"xyz","path":"run.sh","make_executable":true}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::WriteFile {
                make_executable, ..
            } => assert!(make_executable),
            _ => panic!("Expected WriteFile"),
        }
    }

    #[test]
    fn test_deserialize_file_chunk() {
        let json = r#"{"action":"file_chunk","op_id":"abc123","data":"SGVsbG8="}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::FileChunk { op_id, data } => {
                assert_eq!(op_id, "abc123");
                assert_eq!(data, "SGVsbG8=");
            }
            _ => panic!("Expected FileChunk"),
        }
    }

    #[test]
    fn test_deserialize_file_end() {
        let json = r#"{"action":"file_end","op_id":"abc123"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::FileEnd { op_id } => assert_eq!(op_id, "abc123"),
            _ => panic!("Expected FileEnd"),
        }
    }

    #[test]
    fn test_deserialize_warm_repl() {
        let json = r#"{"action":"warm_repl","language":"python"}"#;
        let cmd: GuestCommand = serde_json::from_str(json).unwrap();
        match cmd {
            GuestCommand::WarmRepl { language } => assert_eq!(language, Language::Python),
            _ => panic!("Expected WarmRepl"),
        }
    }

    #[test]
    fn test_deserialize_warm_repl_unknown_language() {
        let json = r#"{"action":"warm_repl","language":"cobol"}"#;
        assert!(serde_json::from_str::<GuestCommand>(json).is_err());
    }

    #[test]
    fn test_deserialize_warm_repl_missing_language() {
        let json = r#"{"action":"warm_repl"}"#;
        assert!(serde_json::from_str::<GuestCommand>(json).is_err());
    }

    // -------------------------------------------------------------------------
    // Response serialization
    // -------------------------------------------------------------------------

    #[test]
    fn test_serialize_file_write_ack() {
        let ack = GuestResponse::FileWriteAck {
            path: "test.txt".to_string(),
            bytes_written: 42,
        };
        let json = serde_json::to_string(&ack).unwrap();
        assert!(json.contains("\"type\":\"file_write_ack\""));
        assert!(json.contains("\"bytes_written\":42"));
    }

    #[test]
    fn test_serialize_file_chunk_response() {
        let chunk = GuestResponse::FileChunk {
            data: "SGVsbG8=".to_string(),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"type\":\"file_chunk\""));
    }

    #[test]
    fn test_serialize_file_read_complete() {
        let msg = GuestResponse::FileReadComplete {
            path: "data.csv".to_string(),
            size: 1024,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"file_read_complete\""));
        assert!(json.contains("\"size\":1024"));
    }

    #[test]
    fn test_serialize_file_list() {
        let list = GuestResponse::FileList {
            entries: vec![
                FileEntry {
                    name: "file.txt".to_string(),
                    is_dir: false,
                    size: 100,
                },
                FileEntry {
                    name: "subdir".to_string(),
                    is_dir: true,
                    size: 0,
                },
            ],
        };
        let json = serde_json::to_string(&list).unwrap();
        assert!(json.contains("\"type\":\"file_list\""));
        assert!(json.contains("\"file.txt\""));
    }
}

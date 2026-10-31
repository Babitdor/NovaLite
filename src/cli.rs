//! Command-line arguments.
//!
//! Parsing lives here rather than in `main.rs` so it can be tested. The
//! binary is a thin shell over the library, and an argument parser that is
//! only reachable by running the program is one that never gets a test.
//!
//! The rules are the ones every command-line tool follows, and each of them
//! is here because the alternative is a confusing failure:
//!
//! * `--help` and `--version` win wherever they appear, so `nova-lite --help`
//!   is not read as a request to use a directory called `--help`.
//! * An unrecognised option is an error, not a directory name. Before this,
//!   `nova-lite --versoin` tried to canonicalise a path called `--versoin`
//!   and reported that it could not be used as the workspace.
//! * `--` ends the options, so a directory whose name starts with a dash can
//!   still be named.

use std::ffi::OsString;
use std::path::PathBuf;

/// What the command line asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Run the TUI.
    ///
    /// `dir` is the directory the agent may touch. `None` means the current
    /// directory, which the caller resolves: parsing touches no filesystem,
    /// so it can be tested without one.
    ///
    /// `auto_approve` is the `--auto-approve` flag: it skips the per-action
    /// prompt, not the workspace boundary or any other limit.
    Run {
        dir: Option<PathBuf>,
        auto_approve: bool,
    },
    /// Print the usage and exit.
    Help,
    /// Print the version and exit.
    Version,
}

/// Parse the arguments after the program name.
pub fn parse_args(args: &[OsString]) -> Result<Command, String> {
    let mut dir: Option<PathBuf> = None;
    let mut auto_approve = false;
    let mut options_ended = false;

    for arg in args {
        let text = arg.to_string_lossy();

        if !options_ended {
            match text.as_ref() {
                "-h" | "--help" => return Ok(Command::Help),
                "-V" | "--version" => return Ok(Command::Version),
                // `--yolo` is the alias: it is what people type, and a flag
                // that is refused because it is spelled the way everyone
                // spells it is a flag that gets worked around with an
                // environment variable instead.
                "--auto-approve" | "--yolo" => {
                    auto_approve = true;
                    continue;
                }
                "--" => {
                    options_ended = true;
                    continue;
                }
                _ => {}
            }
            if text.starts_with('-') {
                return Err(format!("unknown option '{text}'. Try `nova-lite --help`."));
            }
        }

        if dir.is_some() {
            return Err(format!(
                "unexpected argument '{text}': only one directory may be named. \
                 Try `nova-lite --help`."
            ));
        }
        dir = Some(PathBuf::from(arg));
    }

    Ok(Command::Run { dir, auto_approve })
}

/// The version line, printed by `--version`.
pub fn version() -> String {
    format!("nova-lite {}", env!("CARGO_PKG_VERSION"))
}

/// The usage text, printed by `--help`.
///
/// It is the whole interface in one place: the argument, the keys, and the
/// environment. The key list is the short form; `/help` inside the TUI is
/// the long one, and the two are kept in step by hand.
pub fn usage() -> String {
    format!(
        "\
{version}
A coding agent in the terminal.

USAGE:
    nova-lite [OPTIONS] [DIRECTORY]

ARGS:
    DIRECTORY
            The directory the agent may read and write. Defaults to the
            current directory. The first run in a directory asks for
            approval before anything is read or written, and the approval
            is recorded per directory.

OPTIONS:
    -h, --help      Print this help and exit
    -V, --version   Print the version and exit
    --auto-approve  Do not ask before each write or command (alias: --yolo).
                    The workspace boundary, the command timeout and every
                    other limit still apply; only the question is skipped.

KEYS:
    Enter           send the prompt (Shift+Enter inserts a newline)
    y / n           approve or deny the action being asked about
    Esc, Ctrl+C     stop the run in flight; again to quit
    Up, Down        previous prompts
    PageUp/Down     scroll the transcript (Ctrl+Up/Down one line)
    Ctrl+Home/End   the transcript's top / back to following the tail
    Left, Right     move the caret (Ctrl+Left/Right by word)
    Home, End       the caret's line
    Ctrl+A/E/U/K/W  readline editing
    Ctrl+T          expand the newest tool call (Ctrl+Shift+T all)
    Ctrl+G          fold the newest tool group (Ctrl+Shift+G all)
    Tab             complete a /command
    /help           the full list, including the tools

ENVIRONMENT:
    LLM_PROVIDER          ollama (default), anthropic, or openai
    LLM_MODEL             the model to use
    OLLAMA_BASE_URL       the daemon to talk to (default http://localhost:11434)
    LLM_NUM_CTX           context window to request, in tokens (default 262144)
    LLM_MAX_TURNS         model calls allowed per prompt (default 100)
    LLM_COMPACT_AT        context percentage that triggers compaction (default 80)
    LLM_TOOL_RESULT_CHARS character budget for one tool result
    LLM_EVICT_CHARS       character budget for the tool results in one request
    NOVA_APPROVAL         `auto` to skip the per-action prompt, as --auto-approve
    RUST_LOG              log filter; logs go to a file, never the terminal
",
        version = version()
    )
}

//! The tools the model can call.
//!
//! Every tool holds an [`Arc<Runtime>`] and reaches the workspace, the
//! process runner, the approval gate and the budgets through it. There is no
//! constructor that leaves the runtime out, so a tool cannot be built that
//! bypasses the boundary: the type system is what enforces it, not a
//! convention.
//!
//! The tools are the only place the model can change anything, so this is
//! where the checks live. A refusal is returned as a tool error, which rig
//! hands back to the model as the result of the call: the model sees why it
//! was refused and can choose differently, rather than the run failing.

use std::sync::Arc;

use rig::tool::{Tool, ToolContext};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::Error;
use crate::patch::Block;
use crate::runtime::Runtime;
use crate::search;

/// Room reserved for the elision marker, so head + marker + tail still fits
/// the budget the caller asked for.
const MARKER_ALLOWANCE: usize = 200;

/// The opening of the notice [`elide`] leaves where it cut content out.
///
/// The transcript reads a result to decide whether to say it was cut, and the
/// notice is the only thing that survives the trip from the tool to the
/// screen. A constant rather than a literal in two places, with a test pinning
/// it to the text `elide` actually writes.
pub const ELISION_MARKER: &str = "... [elided ";

/// Elide `text` to `budget` characters, keeping the head and the tail.
///
/// The signal in a tool result sits at both ends: a file's imports and its
/// last function, a build log's first error and its final summary line.
/// Keeping only the head would throw away the error at the bottom of a
/// failing test run, which is the most valuable line in the result.
///
/// Counted in `char`s, not bytes: slicing a multi-byte glyph at a byte offset
/// panics, and these results are full of them.
///
/// The result is at most `budget` characters, except that the marker is a
/// floor of roughly [`MARKER_ALLOWANCE`] characters: a budget smaller than
/// that cannot be met without dropping the notice that content was dropped,
/// which is the one thing the model must not be left guessing about.
pub fn elide(text: &str, budget: usize) -> String {
    let total = text.chars().count();
    if total <= budget {
        return text.to_string();
    }

    // The marker is a fixed cost that the head and tail share the budget
    // with. Its length depends on the numbers it reports, which depend on the
    // split, so the split is settled by shrinking it until the whole result
    // fits. `keep` strictly decreases, so this terminates; in practice the
    // first correction is the only one.
    let mut keep = budget.saturating_sub(MARKER_ALLOWANCE);
    loop {
        let head = keep / 2;
        let tail = keep - head;
        let elided = total - keep;

        let head_text: String = text.chars().take(head).collect();
        let tail_text: String = text.chars().skip(total - tail).collect();

        let out = format!(
            "{head_text}\n\n{ELISION_MARKER}{elided} of {total} characters; the first {head} and \
             last {tail} are shown. Narrow the request (grep for the part you need) to see \
             the rest.] ...\n\n{tail_text}"
        );

        let len = out.chars().count();
        if len <= budget || keep == 0 {
            return out;
        }
        keep = keep.saturating_sub(len - budget);
    }
}

/// Deserialize `EditFileArgs::edits` from any of the four shapes a small model
/// actually emits.
///
/// The published schema advertises an array of `{old_string, new_string}`
/// objects, and that is what a well-behaved model sends. Small models send
/// three other things, all of which mean the same:
///
/// 1. the array of objects (the schema),
/// 2. a single bare object, with no array around it,
/// 3. a JSON *string* holding either of the above, double-encoded,
/// 4. a flattened single edit, where `old_string`/`new_string` sit at the top
///    level beside `path` (handled by [`EditFileArgsRaw`], which is why this
///    function only sees shapes 1 to 3).
///
/// Widening what is accepted here costs nothing: the published schema is
/// unchanged, so a model that follows it is unaffected, and a model that does
/// not gets its edit applied instead of burning two model calls on a
/// deserialization error.
fn edits_from_value(value: serde_json::Value) -> Result<Vec<EditBlockArgs>, String> {
    match value {
        // 1. The schema: an array of blocks.
        serde_json::Value::Array(items) => items.into_iter().map(block_from_value).collect(),
        // 2. A single bare block, or 4. a flattened edit. `EditBlockArgs`
        // ignores the fields it does not know, so a flattened edit carrying
        // `path` alongside the two strings deserializes as a block.
        serde_json::Value::Object(_) => Ok(vec![block_from_value(value)?]),
        // 3. A double-encoded string: parse it and try again.
        serde_json::Value::String(text) => {
            let inner: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| format!("`edits` is a string that is not JSON: {e}"))?;
            edits_from_value(inner)
        }
        other => Err(format!(
            "`edits` must be an array of edit blocks, a single edit block, or a JSON string \
             holding either; got {}",
            json_kind(&other)
        )),
    }
}

/// One edit block, from a `{old_string, new_string}` object or a flattened
/// edit that also carries the file's other fields.
fn block_from_value(value: serde_json::Value) -> Result<EditBlockArgs, String> {
    serde_json::from_value(value).map_err(|e| format!("not an edit block: {e}"))
}

/// The name of a JSON value's type, for an error message.
fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Names of all registered tools, in display order.
pub const TOOL_NAMES: &str = "read_file, write_file, edit_file, list_dir, grep, glob, shell";

/// What a tool call is for.
///
/// The transcript groups consecutive calls by this, so a run of eight reads
/// reads as one `EXPLORE` block rather than as eight unrelated rows. The
/// grouping is by *effect*, not by tool: `read_file` and `grep` are the same
/// kind of act, and a reader scanning a long run wants to know whether the
/// agent was looking or changing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCategory {
    /// Reading the project. Nothing on disk changes.
    Explore,
    /// Writing to the project.
    Edit,
    /// Running a command.
    Run,
    /// A tool this build does not know about.
    Other,
}

impl ToolCategory {
    /// The category a tool belongs to, by name.
    ///
    /// An unknown name is [`ToolCategory::Other`] rather than a panic: a tool
    /// added to the agent but not to this table should still render.
    pub fn of(tool: &str) -> Self {
        match tool {
            "read_file" | "list_dir" | "grep" | "glob" => Self::Explore,
            "write_file" | "edit_file" => Self::Edit,
            "shell" => Self::Run,
            _ => Self::Other,
        }
    }

    /// The heading shown above a group of calls in this category.
    pub fn label(self) -> &'static str {
        match self {
            Self::Explore => "EXPLORE",
            Self::Edit => "EDIT",
            Self::Run => "RUN",
            Self::Other => "OTHER",
        }
    }
}

/// Error type for all tools. Rig requires `std::error::Error`; `String` does
/// not implement it, so tool failures are wrapped here.
///
/// The message is what the model reads, so it is built from the typed
/// [`Error`]'s `Display`: a refusal reads the same wherever it is raised.
///
/// The kind is kept alongside the message because rig's dispatch boundary
/// needs it. Its default treatment of an arbitrary error is to keep the text
/// for the operator and show the model a kind-level phrase instead, so a
/// message written *for* the model ("read it again before editing", "that path
/// is outside the workspace") would arrive as "the tool failed". Those
/// messages are the whole of what the model has to act on, so
/// [`ToolError::into_tool_error`] carries them through.
#[derive(Debug)]
pub struct ToolError {
    /// The text the model reads.
    message: String,
    /// How the failure should be presented, and whether it may be retried.
    kind: FailureKind,
}

/// The three ways a tool call can end without a result.
///
/// A refusal is not a failure: the tool worked, the user said no. The
/// distinction is carried all the way to the transcript, which paints them
/// differently, and to the model, which must not re-issue a call the user
/// declined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// The tool could not do what was asked.
    Failed,
    /// The user refused the action.
    Denied,
    /// The run was cancelled while the action was in flight.
    Cancelled,
}

impl ToolError {
    /// A failure with `message` as the text the model reads.
    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: FailureKind::Failed,
        }
    }

    /// The text the model reads.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// How the call ended.
    pub fn kind(&self) -> FailureKind {
        self.kind
    }

    /// Normalize for rig's dispatch boundary.
    ///
    /// rig's default is [`rig::tool::ToolExecutionError::from_error`], which
    /// treats an arbitrary error as operator-only and replaces the model's
    /// view of it with a phrase chosen by kind. Every message a tool writes
    /// for the model would arrive as "the tool failed", which is both
    /// unactionable and, for a denial, untrue. The message is carried through
    /// instead, and the kind is preserved so a refusal stays a refusal: rig
    /// marks it non-retryable and hooks can see it as a decision rather than a
    /// fault.
    pub fn into_tool_error(self) -> rig::tool::ToolExecutionError {
        match self.kind {
            FailureKind::Failed => rig::tool::ToolExecutionError::other(self.message),
            FailureKind::Denied => rig::tool::ToolExecutionError::refused(format!(
                "{}. The user declined this action; do not repeat it.",
                self.message
            )),
            FailureKind::Cancelled => rig::tool::ToolExecutionError::cancelled(self.message),
        }
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ToolError {}

impl From<Error> for ToolError {
    fn from(err: Error) -> Self {
        let kind = match err {
            Error::Denied(_) => FailureKind::Denied,
            Error::Cancelled => FailureKind::Cancelled,
            _ => FailureKind::Failed,
        };
        ToolError {
            message: err.to_string(),
            kind,
        }
    }
}

/// The runtime every tool is built with.
///
/// A type alias rather than a bare `Arc<Runtime>` so the tool structs read as
/// "this tool needs the session", which is the point of the change.
pub type Session = Arc<Runtime>;

// ---------------------------------------------------------------------------
// read_file
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ReadFileArgs {
    #[serde(
        alias = "file_path",
        alias = "filepath",
        alias = "filename",
        alias = "file",
        alias = "dir",
        alias = "directory"
    )]
    pub path: String,
    /// The first line to return, counting from 1.
    #[serde(default, alias = "from_line", alias = "start")]
    pub start_line: Option<usize>,
    /// The last line to return, counting from 1 and inclusive.
    #[serde(default, alias = "to_line", alias = "end")]
    pub end_line: Option<usize>,
}

pub struct ReadFile {
    rt: Session,
}

impl ReadFile {
    /// A `read_file` bound to `rt`.
    pub fn new(rt: Session) -> Self {
        Self { rt }
    }
}

impl Tool for ReadFile {
    const NAME: &'static str = "read_file";
    type Args = ReadFileArgs;
    type Output = String;
    type Error = ToolError;

    fn map_error(&self, error: ToolError) -> rig::tool::ToolExecutionError {
        error.into_tool_error()
    }

    fn description(&self) -> String {
        "Read the contents of a file at the given path, relative to the workspace root. \
         Reading a file is also what lets edit_file change it: an edit is refused if the \
         file has changed since you last read it, so read it again after such a refusal. \
         Pass start_line and end_line to read one part of a long file."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to read, relative to the workspace root" },
                "start_line": { "type": "integer", "description": "First line to return, counting from 1. Defaults to the first line." },
                "end_line": { "type": "integer", "description": "Last line to return, counting from 1 and inclusive. Defaults to the last line." }
            },
            "required": ["path"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        // The read goes through the session's service, which resolves the
        // path against the real filesystem target and remembers what the file
        // held. The remembering is what lets a later edit refuse to apply to a
        // file this session has not actually seen.
        let read = self.rt.read_file(&args.path).await?;

        // A line range is how the model gets at the part of a file the budget
        // cut off: `grep` reports a match on line 4,000, and this is how it
        // reads around that line without pulling the other 3,999 through the
        // context. The digest the service took is over the whole file either
        // way, so reading a range still counts as having read the file.
        let (text, note) = match (args.start_line, args.end_line) {
            (None, None) => (read.content.clone(), None),
            (start, end) => {
                let start = start.unwrap_or(1).max(1);
                let end = end.unwrap_or(usize::MAX);
                if end < start {
                    return Err(Error::invalid(format!(
                        "end_line ({end}) is before start_line ({start})"
                    ))
                    .into());
                }
                let lines: Vec<&str> = read.content.lines().collect();
                let total = lines.len();
                let from = (start - 1).min(total);
                let to = end.min(total);
                let note = if to <= from {
                    format!("\n\n... [no lines in that range; the file has {total} lines] ...")
                } else {
                    format!("\n\n... [lines {}-{} of {total}] ...", from + 1, to)
                };
                (lines[from..to].join("\n"), Some(note))
            }
        };

        // The budget is the only cap: a line count is a proxy for "do not
        // blow the context", and the budget measures that directly, in the
        // unit the context is measured in.
        let mut out = elide(&text, self.rt.tool_result_budget());
        if let Some(note) = note {
            out.push_str(&note);
        }
        if let Some(size) = read.truncated_from {
            out.push_str(&format!(
                "\n\n... [the file is {size} bytes; only the first {} were read. Use grep to \
                 find the part you need.] ...",
                self.rt.config().limits.max_read_bytes
            ));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// write_file
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct WriteFileArgs {
    #[serde(
        alias = "file_path",
        alias = "filepath",
        alias = "filename",
        alias = "file",
        alias = "dir",
        alias = "directory"
    )]
    pub path: String,
    #[serde(alias = "contents", alias = "text", alias = "data")]
    pub content: String,
}

pub struct WriteFile {
    rt: Session,
}

impl WriteFile {
    /// A `write_file` bound to `rt`.
    pub fn new(rt: Session) -> Self {
        Self { rt }
    }
}

impl Tool for WriteFile {
    const NAME: &'static str = "write_file";
    type Args = WriteFileArgs;
    type Output = String;
    type Error = ToolError;

    fn map_error(&self, error: ToolError) -> rig::tool::ToolExecutionError {
        error.into_tool_error()
    }

    fn description(&self) -> String {
        "Write content to a file at the given path, creating parent directories if needed. The path is relative to the workspace root.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to write, relative to the workspace root" },
                "content": { "type": "string", "description": "Full content to write to the file" }
            },
            "required": ["path", "content"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        // The write goes through the session's execution service, which
        // resolves the path, asks for approval and re-checks the file's
        // version before it commits. The tool does not touch the filesystem
        // itself: a tool that could would be a second way in, and the gate
        // would only cover the paths that remembered to use it.
        self.rt.write_file(&args.path, &args.content).await?;
        Ok(elide(
            &format!(
                "Successfully wrote {} bytes to {}",
                args.content.len(),
                args.path
            ),
            self.rt.tool_result_budget(),
        ))
    }
}

// ---------------------------------------------------------------------------
// edit_file
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(try_from = "EditFileArgsRaw")]
pub struct EditFileArgs {
    #[serde(
        alias = "file_path",
        alias = "filepath",
        alias = "filename",
        alias = "file",
        alias = "dir",
        alias = "directory"
    )]
    pub path: String,
    /// The replacements to apply, in order.
    ///
    /// Deserialized leniently: see [`edits_from_value`] for the four shapes a
    /// small model actually emits.
    pub edits: Vec<EditBlockArgs>,
}

/// The wire form of [`EditFileArgs`], as permissive as a small model needs.
///
/// A separate type because the fourth shape a model emits puts the edit's two
/// strings at the top level beside `path`, and a field-level deserializer only
/// ever sees its own field's value: it cannot know that `old_string` was there.
/// Deserializing the whole object first is what makes that shape reachable.
#[derive(Debug, Deserialize)]
struct EditFileArgsRaw {
    #[serde(
        alias = "file_path",
        alias = "filepath",
        alias = "filename",
        alias = "file",
        alias = "dir",
        alias = "directory"
    )]
    path: String,
    #[serde(default, alias = "changes", alias = "replacements")]
    edits: Option<serde_json::Value>,
    #[serde(
        default,
        alias = "old",
        alias = "old_str",
        alias = "search",
        alias = "find"
    )]
    old_string: Option<String>,
    #[serde(
        default,
        alias = "new",
        alias = "new_str",
        alias = "replace",
        alias = "replacement"
    )]
    new_string: Option<String>,
    #[serde(default)]
    replace_all: bool,
}

impl TryFrom<EditFileArgsRaw> for EditFileArgs {
    type Error = String;

    fn try_from(raw: EditFileArgsRaw) -> Result<Self, Self::Error> {
        let edits = match (raw.edits, raw.old_string, raw.new_string) {
            // Shapes 1 to 3: `edits` is present, in one of the forms
            // `edits_from_value` accepts.
            (Some(value), _, _) => edits_from_value(value)?,
            // Shape 4: a flattened single edit, with the two strings beside
            // `path` rather than inside a block.
            (None, Some(old_string), Some(new_string)) => vec![EditBlockArgs {
                old_string,
                new_string,
                replace_all: raw.replace_all,
            }],
            (None, _, _) => {
                return Err(
                    "`edits` is required: send an array of `{old_string, new_string}` blocks, \
                     or `old_string` and `new_string` beside `path` for a single edit."
                        .to_string(),
                )
            }
        };
        Ok(Self {
            path: raw.path,
            edits,
        })
    }
}

/// One search-and-replace block.
#[derive(Debug, Deserialize)]
pub struct EditBlockArgs {
    /// The exact text to find. Must not be empty.
    #[serde(alias = "old", alias = "old_str", alias = "search", alias = "find")]
    pub old_string: String,
    /// What to put in its place. May be empty, which deletes.
    #[serde(
        alias = "new",
        alias = "new_str",
        alias = "replace",
        alias = "replacement"
    )]
    pub new_string: String,
    /// Replace every occurrence instead of requiring exactly one.
    #[serde(default)]
    pub replace_all: bool,
}

impl From<&EditBlockArgs> for Block {
    fn from(args: &EditBlockArgs) -> Self {
        Self {
            old: args.old_string.clone(),
            new: args.new_string.clone(),
            replace_all: args.replace_all,
        }
    }
}

pub struct EditFile {
    rt: Session,
}

impl EditFile {
    /// An `edit_file` bound to `rt`.
    pub fn new(rt: Session) -> Self {
        Self { rt }
    }
}

impl Tool for EditFile {
    const NAME: &'static str = "edit_file";
    type Args = EditFileArgs;
    type Output = String;
    type Error = ToolError;

    fn map_error(&self, error: ToolError) -> rig::tool::ToolExecutionError {
        error.into_tool_error()
    }

    fn description(&self) -> String {
        "Change an existing file by replacing exact text in it. Each edit's old_string must \
         match the file exactly and, unless replace_all is set, must match in exactly one \
         place; add surrounding lines to make it unique. Edits are applied in order, each \
         against the result of the one before, and the file is left untouched unless every \
         edit applies. Use write_file to create a file or to replace one whole. The path is \
         relative to the workspace root."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to edit, relative to the workspace root" },
                "edits": {
                    "type": "array",
                    "description": "The replacements to apply, in order. One entry for a single change.",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": { "type": "string", "description": "Exact text to find, copied from the file, including indentation. Must match in exactly one place unless replace_all is set." },
                            "new_string": { "type": "string", "description": "Replacement text. Empty deletes the matched text." },
                            "replace_all": { "type": "boolean", "description": "Replace every occurrence rather than requiring exactly one. Defaults to false." }
                        },
                        "required": ["old_string", "new_string"]
                    }
                }
            },
            "required": ["path", "edits"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let blocks: Vec<Block> = args.edits.iter().map(Block::from).collect();
        // Read, check and write are one operation inside the service: the
        // version the replacement was computed from is the version the commit
        // is checked against, which is only true if nothing runs between
        // them.
        self.rt.edit_file(&args.path, &blocks).await?;
        Ok(elide(
            &format!(
                "Successfully edited '{}' ({} change{})",
                args.path,
                blocks.len(),
                if blocks.len() == 1 { "" } else { "s" }
            ),
            self.rt.tool_result_budget(),
        ))
    }
}

// ---------------------------------------------------------------------------
// shell
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ShellArgs {
    #[serde(alias = "cmd", alias = "script", alias = "shell_command")]
    pub command: String,
}

pub struct Shell {
    rt: Session,
}

impl Shell {
    /// A `shell` bound to `rt`.
    pub fn new(rt: Session) -> Self {
        Self { rt }
    }
}

impl Tool for Shell {
    const NAME: &'static str = "shell";
    type Args = ShellArgs;
    type Output = String;
    type Error = ToolError;

    fn map_error(&self, error: ToolError) -> rig::tool::ToolExecutionError {
        error.into_tool_error()
    }

    fn description(&self) -> String {
        // The shell is named because the model otherwise guesses, and on
        // Windows it guesses bash: `for f in a b; do ...; done` is rejected by
        // `cmd.exe` and the call is wasted. Which shell it is is a property of
        // the build, so the description is too.
        let shell = if cfg!(windows) { "cmd.exe" } else { "sh" };
        format!(
            "Run a shell command and return its combined stdout and stderr. \
             The command runs in the workspace root, through {shell}; write it \
             in {shell} syntax. The working directory is the workspace, but the \
             command itself is not confined to it: prefer the file tools, which \
             are, and use this for building and testing."
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to execute" }
            },
            "required": ["command"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let out = self.rt.run_command(&args.command).await?;
        let limit = self.rt.config().limits.max_shell_bytes;
        Ok(elide(&out.render(limit), self.rt.tool_result_budget()))
    }
}

// ---------------------------------------------------------------------------
// list_dir
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ListDirArgs {
    #[serde(
        alias = "file_path",
        alias = "filepath",
        alias = "filename",
        alias = "file",
        alias = "dir",
        alias = "directory"
    )]
    pub path: String,
}

pub struct ListDir {
    rt: Session,
}

impl ListDir {
    /// A `list_dir` bound to `rt`.
    pub fn new(rt: Session) -> Self {
        Self { rt }
    }
}

impl Tool for ListDir {
    const NAME: &'static str = "list_dir";
    type Args = ListDirArgs;
    type Output = String;
    type Error = ToolError;

    fn map_error(&self, error: ToolError) -> rig::tool::ToolExecutionError {
        error.into_tool_error()
    }

    fn description(&self) -> String {
        "List the contents of a directory, relative to the workspace root.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path to list, relative to the workspace root" }
            },
            "required": ["path"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let dir = self.rt.resolve_existing(&args.path).await?;
        let budget = search::Budget::from_limits(&self.rt.config().limits);
        let stop = self.rt.cancel().clone();

        // A directory with a hundred thousand entries is a real thing to
        // point this at, and building a hundred thousand strings to throw
        // almost all of them away is the failure the bound exists to prevent.
        let out = search::on_worker(move || search::list(&dir, &budget, stop.as_ref())).await?;

        Ok(elide(
            &out.render("(empty directory)"),
            self.rt.tool_result_budget(),
        ))
    }
}

// ---------------------------------------------------------------------------
// grep
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct GrepArgs {
    #[serde(alias = "query", alias = "regex", alias = "search", alias = "glob")]
    pub pattern: String,
    /// The directory to search, relative to the workspace root.
    #[serde(
        default = "workspace_root",
        alias = "file_path",
        alias = "filepath",
        alias = "filename",
        alias = "file",
        alias = "dir",
        alias = "directory"
    )]
    pub path: String,
    /// Also search dot-files and dot-directories.
    #[serde(default)]
    pub hidden: bool,
    /// Also search paths the ignore files exclude.
    #[serde(default)]
    pub ignored: bool,
}

/// The default search root: the whole workspace.
///
/// A search that has to be told the root every time is a search the model
/// forgets to aim, and `grep` over the whole workspace is what it means nine
/// times in ten.
fn workspace_root() -> String {
    ".".to_string()
}

pub struct Grep {
    rt: Session,
}

impl Grep {
    /// A `grep` bound to `rt`.
    pub fn new(rt: Session) -> Self {
        Self { rt }
    }
}

impl Tool for Grep {
    const NAME: &'static str = "grep";
    type Args = GrepArgs;
    type Output = String;
    type Error = ToolError;

    fn map_error(&self, error: ToolError) -> rig::tool::ToolExecutionError {
        error.into_tool_error()
    }

    fn description(&self) -> String {
        "Search file contents for a regex pattern, returning 'path:line: content' matches. \
         The search is recursive and respects .gitignore and .ignore, so build output and \
         vendored trees are skipped; set ignored to search them anyway. Results are capped, \
         and a capped result says so rather than looking complete."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regex pattern to search for" },
                "path": { "type": "string", "description": "Directory to search, relative to the workspace root. Defaults to the whole workspace." },
                "hidden": { "type": "boolean", "description": "Also search dot-files and dot-directories. Defaults to false." },
                "ignored": { "type": "boolean", "description": "Also search paths excluded by .gitignore or .ignore. Defaults to false." }
            },
            "required": ["pattern"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        // The path is resolved here, on the async side, so a refusal is
        // reported before a worker is spawned for a search that cannot run.
        let root = self.rt.resolve_existing(&args.path).await?;
        let pattern = args.pattern.clone();
        let opts = search::Options {
            hidden: args.hidden,
            ignored: args.ignored,
            max_depth: None,
        };
        let budget = search::Budget::from_limits(&self.rt.config().limits);
        let stop = self.rt.cancel().clone();

        // The walk and the matching are CPU-bound and have no await point to
        // yield at, so they run on a blocking worker rather than stalling the
        // runtime for the length of the search.
        let out =
            search::on_worker(move || search::grep(&root, &pattern, &opts, &budget, stop.as_ref()))
                .await?;

        Ok(elide(
            &out.render(&format!("No matches found for '{}'", args.pattern)),
            self.rt.tool_result_budget(),
        ))
    }
}

// ---------------------------------------------------------------------------
// glob
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct GlobArgs {
    #[serde(alias = "query", alias = "regex", alias = "search", alias = "glob")]
    pub pattern: String,
    /// The directory to search, relative to the workspace root.
    #[serde(
        default = "workspace_root",
        alias = "file_path",
        alias = "filepath",
        alias = "filename",
        alias = "file",
        alias = "dir",
        alias = "directory"
    )]
    pub path: String,
    /// Also match dot-files and dot-directories.
    #[serde(default)]
    pub hidden: bool,
    /// Also match paths the ignore files exclude.
    #[serde(default)]
    pub ignored: bool,
}

pub struct Glob {
    rt: Session,
}

impl Glob {
    /// A `glob` bound to `rt`.
    pub fn new(rt: Session) -> Self {
        Self { rt }
    }
}

impl Tool for Glob {
    const NAME: &'static str = "glob";
    type Args = GlobArgs;
    type Output = String;
    type Error = ToolError;

    fn map_error(&self, error: ToolError) -> rig::tool::ToolExecutionError {
        error.into_tool_error()
    }

    fn description(&self) -> String {
        "Find files and directories whose paths match a glob pattern (e.g. '**/*.rs'), \
         searched from the workspace root. The search respects .gitignore and .ignore; set \
         ignored to search them anyway. Results are capped, and a capped result says so."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern, e.g. '**/*.rs', matched against the path relative to the workspace root" },
                "path": { "type": "string", "description": "Directory to search, relative to the workspace root. Defaults to the whole workspace." },
                "hidden": { "type": "boolean", "description": "Also match dot-files and dot-directories. Defaults to false." },
                "ignored": { "type": "boolean", "description": "Also match paths excluded by .gitignore or .ignore. Defaults to false." }
            },
            "required": ["pattern"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let root = self.rt.resolve_existing(&args.path).await?;
        let pattern = args.pattern.clone();
        let opts = search::Options {
            hidden: args.hidden,
            ignored: args.ignored,
            max_depth: None,
        };
        let budget = search::Budget::from_limits(&self.rt.config().limits);
        let stop = self.rt.cancel().clone();

        let out =
            search::on_worker(move || search::glob(&root, &pattern, &opts, &budget, stop.as_ref()))
                .await?;

        Ok(elide(
            &out.render(&format!("No files matched '{}'", args.pattern)),
            self.rt.tool_result_budget(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ApprovalMode, Config, DEFAULT_MAX_SEARCH_MATCHES};
    use crate::workspace::Workspace;

    /// A runtime over the temp directory, approving everything: these tests
    /// are about the tools, not the gate.
    fn rt() -> Session {
        let ws = Workspace::new(std::env::temp_dir()).expect("the temp dir is a valid root");
        Runtime::new(Config::defaults().with_approval(ApprovalMode::Auto), ws)
    }

    /// Helper: run a tool call with a throwaway context.
    macro_rules! call_tool {
        ($tool:expr, $args:expr) => {{
            let mut ctx = ToolContext::default();
            $tool.call(&mut ctx, $args).await
        }};
    }

    /// The workspace root for these tests: the temp directory, so the
    /// absolute temp paths below are inside it.
    fn test_root() -> std::path::PathBuf {
        rt().workspace().root().to_path_buf()
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        test_root().join(format!("rust_agent_test_{}_{}", std::process::id(), name))
    }

    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let path = tmp_path("roundtrip.txt");
        let path_str = path.display().to_string();
        let _ = std::fs::remove_file(&path);

        let write = call_tool!(
            WriteFile::new(rt()),
            WriteFileArgs {
                path: path_str.clone(),
                content: "hello\nworld\n".to_string()
            }
        );
        assert!(write.is_ok(), "write failed: {:?}", write);

        let read = call_tool!(
            ReadFile::new(rt()),
            ReadFileArgs {
                path: path_str.clone(),
                start_line: None,
                end_line: None,
            }
        )
        .expect("read failed");
        assert_eq!(read, "hello\nworld\n");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn edit_file_replaces_the_one_occurrence() {
        let path = tmp_path("edit.txt");
        let path_str = path.display().to_string();
        std::fs::write(&path, "a b c\n").unwrap();

        let res = call_tool!(
            EditFile::new(rt()),
            EditFileArgs {
                path: path_str.clone(),
                edits: vec![EditBlockArgs {
                    old_string: "b".to_string(),
                    new_string: "X".to_string(),
                    replace_all: false,
                }]
            }
        );
        assert!(res.is_ok(), "edit failed: {:?}", res);

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "a X c\n");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn edit_file_refuses_a_search_that_matches_twice() {
        // The old behaviour replaced the first match and said nothing, which
        // is how an edit lands in the wrong place. Now it is an error that
        // names the lines, so the model can add context.
        let path = tmp_path("edit_ambiguous.txt");
        let path_str = path.display().to_string();
        std::fs::write(&path, "a\nb\na\nb\na\n").unwrap();

        let res = call_tool!(
            EditFile::new(rt()),
            EditFileArgs {
                path: path_str.clone(),
                edits: vec![EditBlockArgs {
                    old_string: "a".to_string(),
                    new_string: "X".to_string(),
                    replace_all: false,
                }]
            }
        );
        let err = res.expect_err("an ambiguous search should be refused");
        let message = err.to_string();
        assert!(
            message.contains("matches 3 places"),
            "unexpected: {message}"
        );
        assert!(message.contains("lines 1, 3, 5"), "unexpected: {message}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "a\nb\na\nb\na\n",
            "a refused edit must leave the file untouched"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn edit_file_applies_several_blocks_in_order() {
        let path = tmp_path("edit_many.txt");
        let path_str = path.display().to_string();
        std::fs::write(&path, "one\ntwo\nthree\n").unwrap();

        let res = call_tool!(
            EditFile::new(rt()),
            EditFileArgs {
                path: path_str.clone(),
                edits: vec![
                    EditBlockArgs {
                        old_string: "one".to_string(),
                        new_string: "1".to_string(),
                        replace_all: false,
                    },
                    EditBlockArgs {
                        old_string: "three".to_string(),
                        new_string: "3".to_string(),
                        replace_all: false,
                    },
                ]
            }
        );
        assert!(res.is_ok(), "edit failed: {:?}", res);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "1\ntwo\n3\n");

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn edit_file_missing_old_string_errors() {
        let path = tmp_path("edit_missing.txt");
        let path_str = path.display().to_string();
        std::fs::write(&path, "abc").unwrap();

        let res = call_tool!(
            EditFile::new(rt()),
            EditFileArgs {
                path: path_str.clone(),
                edits: vec![EditBlockArgs {
                    old_string: "zzz".to_string(),
                    new_string: "y".to_string(),
                    replace_all: false,
                }]
            }
        );
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("not found"));

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn read_file_missing_errors_with_context() {
        let path = tmp_path("does_not_exist.txt");
        let path_str = path.display().to_string();
        let res = call_tool!(
            ReadFile::new(rt()),
            ReadFileArgs {
                path: path_str,
                start_line: None,
                end_line: None,
            }
        );
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("reading"));
    }

    #[tokio::test]
    async fn shell_returns_stdout() {
        let out = call_tool!(
            Shell::new(rt()),
            ShellArgs {
                command: "echo hello_agent".to_string()
            }
        )
        .expect("shell failed");
        assert!(out.contains("hello_agent"), "unexpected output: {out}");
    }

    #[tokio::test]
    async fn shell_reports_a_nonzero_exit() {
        // A failing command is a result, not a tool error: the model needs to
        // see the exit status to decide what to do next.
        let out = call_tool!(
            Shell::new(rt()),
            ShellArgs {
                command: "exit 3".to_string()
            }
        )
        .expect("a nonzero exit should not be a tool error");
        assert!(out.contains("exit code 3"), "unexpected output: {out}");
    }

    #[tokio::test]
    async fn shell_reports_a_nonzero_exit_even_when_the_command_printed_something() {
        // The old result reported the status only when the output was empty,
        // so a failing command that printed anything looked like a success.
        // This is the case that made a broken build read as a passing one.
        let out = call_tool!(
            Shell::new(rt()),
            ShellArgs {
                command: "echo the-build-failed & exit 3".to_string()
            }
        )
        .expect("a nonzero exit should not be a tool error");
        assert!(out.contains("the-build-failed"), "unexpected output: {out}");
        assert!(out.contains("exit code 3"), "unexpected output: {out}");
    }

    #[tokio::test]
    async fn shell_kills_a_command_that_outlives_the_timeout() {
        // A short bound, so the test proves the mechanism rather than waiting
        // out the real 120s. `ping -n 30` runs far longer than 300ms.
        //
        // A timeout is a *result*, not an error: the command ran, it was
        // killed, and whatever it printed before the kill is what the model
        // needs. Discarding it was the old behaviour, and it threw away
        // exactly the diagnostics a hung build had already produced.
        let ws = Workspace::new(std::env::temp_dir()).expect("a valid root");
        let mut config = Config::defaults().with_approval(ApprovalMode::Auto);
        config.command_timeout = std::time::Duration::from_millis(300);
        let rt = Runtime::new(config, ws);

        let started = std::time::Instant::now();
        let out = rt
            .process()
            .run_shell("ping -n 30 127.0.0.1")
            .await
            .expect("a killed command is a result, not a spawn failure");

        assert_eq!(
            out.termination,
            crate::process::Termination::TimedOut,
            "unexpected termination: {:?}",
            out.termination
        );
        assert!(!out.succeeded());
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "the timeout should have fired promptly, took {:?}",
            started.elapsed()
        );
        // The kill reached the whole tree, so the readers saw end-of-stream
        // rather than waiting out the ping.
        assert!(
            out.read_error.is_none(),
            "the pipes should have closed cleanly: {:?}",
            out.read_error
        );
    }

    #[tokio::test]
    async fn shell_under_the_timeout_still_succeeds() {
        // The bound must not break ordinary commands.
        let out = rt()
            .process()
            .run_shell("echo quick")
            .await
            .expect("a fast command should succeed");
        assert!(out.stdout.contains("quick"), "unexpected output: {out:?}");
    }

    #[tokio::test]
    async fn list_dir_lists_entries() {
        let dir = tmp_path("listdir");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "a").unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();

        let out = call_tool!(
            ListDir::new(rt()),
            ListDirArgs {
                path: dir.display().to_string()
            }
        )
        .expect("list_dir failed");
        assert!(out.contains("a.txt"), "unexpected output: {out}");
        assert!(out.contains("sub/"), "unexpected output: {out}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_finds_matches() {
        let dir = tmp_path("grepdir");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("x.txt"), "hello\nworld\nhello again\n").unwrap();

        let out = call_tool!(
            Grep::new(rt()),
            GrepArgs {
                pattern: "hello".to_string(),
                path: dir.display().to_string(),
                hidden: false,
                ignored: false,
            }
        )
        .expect("grep failed");
        assert!(out.contains("x.txt:1:"), "unexpected output: {out}");
        assert!(out.contains("x.txt:3:"), "unexpected output: {out}");
        assert!(!out.contains("world"), "unexpected output: {out}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn glob_matches_files() {
        let dir = tmp_path("globdir");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "").unwrap();
        std::fs::write(dir.join("b.txt"), "").unwrap();

        let out = call_tool!(
            Glob::new(rt()),
            GlobArgs {
                pattern: "**/*.rs".to_string(),
                path: dir.display().to_string(),
                hidden: false,
                ignored: false,
            }
        )
        .expect("glob failed");
        assert!(out.contains("a.rs"), "unexpected output: {out}");
        assert!(!out.contains("b.txt"), "unexpected output: {out}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ------------------------------------------------- the tool-result budget

    /// A directory holding one file whose contents exceed the model's
    /// per-result budget several times over, and the budget it exceeds.
    ///
    /// The budget is read from the same place the tools read it, so the test
    /// adapts to `LLM_TOOL_RESULT_CHARS` / `LLM_NUM_CTX` rather than assuming
    /// the defaults.
    ///
    /// The *line* is sized from the budget as well as the file, because a
    /// search stops at its own match limit: a file of short lines can be many
    /// times the budget and still produce a result well under it, so the cap
    /// is only exercised when the matches a search is allowed to collect are
    /// themselves longer than the budget.
    fn oversized_dir(name: &str) -> (std::path::PathBuf, usize) {
        let budget = rt().tool_result_budget();
        let dir = tmp_path(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Longer than `budget / max_matches`, so the matches a search may
        // collect add up to more than the budget.
        let padding = budget / DEFAULT_MAX_SEARCH_MATCHES + 1;
        let line = format!("needle in a haystack {}\n", "x".repeat(padding));
        let count = budget / line.len() * 3 + 10;
        std::fs::write(dir.join("big.txt"), line.repeat(count)).unwrap();
        (dir, budget)
    }

    #[test]
    fn elide_leaves_a_result_under_the_budget_alone() {
        let text = "short enough";
        assert_eq!(
            elide(text, 1_000),
            text,
            "an under-budget result must be returned untouched"
        );
    }

    #[test]
    fn elide_keeps_the_head_and_the_tail() {
        // The signal is at both ends: a build log's first error and its final
        // summary line. A head-only elision would drop the second.
        let text: String = (0..1_000)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let out = elide(&text, 400);

        assert!(out.starts_with(&text[..10]), "the head must survive: {out}");
        assert!(
            out.ends_with(&text[text.len() - 10..]),
            "the tail must survive: {out}"
        );
        assert!(
            out.contains("elided"),
            "the model must be told content was dropped: {out}"
        );
    }

    #[test]
    fn elide_never_exceeds_the_budget() {
        // Asserted at both ends of the range: a `<=` guard would pass the over
        // case while letting the under case through, and vice versa.
        let text = "x".repeat(10_000);
        for budget in [200, 201, 500, 1_000, 4_096] {
            let out = elide(&text, budget);
            assert!(
                out.chars().count() <= budget,
                "budget {budget} produced {} characters",
                out.chars().count()
            );
        }

        // The boundary itself: exactly at the budget is not over it.
        let exact = "y".repeat(500);
        assert_eq!(
            elide(&exact, 500),
            exact,
            "a result exactly at the budget is not over it"
        );
        assert!(
            elide(&exact, 499).contains("elided"),
            "one character over the budget is elided"
        );
    }

    #[test]
    fn elide_does_not_split_a_multibyte_glyph() {
        // Slicing at a byte offset panics on these, and tool results are full
        // of them.
        let text = "→é".repeat(2_000);
        let out = elide(&text, 300);

        assert!(
            out.chars().count() <= 300,
            "got {} characters",
            out.chars().count()
        );
        assert!(
            out.starts_with('→'),
            "the head must start on a glyph boundary"
        );
        assert!(out.ends_with('é'), "the tail must end on a glyph boundary");
    }

    #[test]
    fn elide_survives_a_budget_smaller_than_its_own_marker() {
        // The marker is a floor: a budget below it cannot be met without
        // dropping the notice that content was dropped, which is the one
        // thing the model must not be left guessing about. It must still not
        // panic or underflow.
        let text = "z".repeat(10_000);
        for budget in [0, 1, 10, 50, 199] {
            let out = elide(&text, budget);
            assert!(
                out.contains("elided"),
                "budget {budget} lost the marker: {out}"
            );
        }
    }

    #[tokio::test]
    async fn read_file_caps_a_file_larger_than_the_budget() {
        let (dir, budget) = oversized_dir("read_cap");

        let out = call_tool!(
            ReadFile::new(rt()),
            ReadFileArgs {
                path: dir.join("big.txt").display().to_string(),
                start_line: None,
                end_line: None,
            }
        )
        .expect("read failed");

        assert!(
            out.chars().count() <= budget,
            "read_file returned {} characters against a {budget} budget",
            out.chars().count()
        );
        assert!(out.contains("elided"), "the elision must be visible: {out}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_caps_a_result_larger_than_the_budget() {
        let (dir, budget) = oversized_dir("grep_cap");

        let out = call_tool!(
            Grep::new(rt()),
            GrepArgs {
                pattern: "needle".to_string(),
                path: dir.display().to_string(),
                hidden: false,
                ignored: false,
            }
        )
        .expect("grep failed");

        assert!(
            out.chars().count() <= budget,
            "grep returned {} characters against a {budget} budget",
            out.chars().count()
        );
        assert!(out.contains("elided"), "the elision must be visible: {out}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn shell_passes_a_quoted_argument_through_unchanged() {
        // `cmd.exe` parses the raw command line itself; it does not use the
        // MSVC quoting rules `Command::arg` applies. Passing the command
        // through `arg` therefore mangles anything containing a quote:
        // `echo "a b"` arrives as `echo \"a b\"` and prints the backslashes.
        let out = call_tool!(
            Shell::new(rt()),
            ShellArgs {
                command: "echo \"hello world\"".to_string()
            }
        )
        .expect("shell failed");

        assert!(
            out.contains("hello world"),
            "the quoted text should survive: {out:?}"
        );
        assert!(
            !out.contains('\\'),
            "the quotes should not be escaped with backslashes: {out:?}"
        );
    }

    #[tokio::test]
    async fn shell_caps_a_result_larger_than_the_budget() {
        let (dir, budget) = oversized_dir("shell_cap");

        let out = call_tool!(
            Shell::new(rt()),
            ShellArgs {
                // Quoted, because the path is quoted in real use and the
                // quoting is what used to break: `cmd /C` needs the command
                // verbatim, and Rust's own argument quoting escaped the
                // quotes with backslashes that `cmd` printed literally.
                command: format!("type \"{}\"", dir.join("big.txt").display())
            }
        )
        .expect("shell failed");

        assert!(
            out.chars().count() <= budget,
            "shell returned {} characters against a {budget} budget",
            out.chars().count()
        );
        assert!(out.contains("elided"), "the elision must be visible: {out}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn shell_runs_a_quoted_path_containing_a_space() {
        // The case the quoting bug actually broke: a path with a space in it
        // has to be quoted, and the quotes have to reach `cmd` intact.
        let dir = tmp_path("shell space");
        std::fs::create_dir_all(&dir).expect("creating the directory");
        let file = dir.join("a file.txt");
        std::fs::write(&file, "contents\n").expect("writing the file");

        let out = call_tool!(
            Shell::new(rt()),
            ShellArgs {
                command: format!("type \"{}\"", file.display())
            }
        )
        .expect("shell failed");

        assert!(
            out.contains("contents"),
            "a quoted path with a space should be read: {out:?}"
        );
        assert!(
            !out.contains("cannot find"),
            "the path should have been found: {out:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}

//! The services a tool call runs against.
//!
//! A tool used to reach for process-global state: the workspace root was a
//! `OnceLock`, the timeout was a `const`, and the approval gate did not exist.
//! That made two sessions in one process impossible, made tests
//! order-dependent, and left no single place to enforce a boundary.
//!
//! [`Runtime`] is that place. It is built once, at startup, from a
//! [`Config`] and a [`Workspace`], and handed to every tool. A tool cannot be
//! constructed without one, so there is no unrestricted path left to take by
//! accident.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::approval::{
    diff_preview, new_file_preview, ActionKind, ApprovalRequest, Approver, AutoApprover, Decision,
    NoChannel,
};
use crate::config::Config;
use crate::context::Estimate;
use crate::error::{Error, Result};
use crate::filesystem::{self, read_head, Version};
use crate::patch::{self, Block};
use crate::process::{CommandOutput, ProcessRunner};
use crate::workspace::{Reach, Workspace};

/// How often each repair fired during the session.
///
/// The TUI does not show these; the eval harness does. A run that fails is
/// otherwise a single opaque outcome, and "the model kept naming tools that do
/// not exist" and "the daemon stalled" call for completely different fixes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RepairReport {
    /// Tool names repaired to a registered tool.
    pub tool_name_repairs: usize,
    /// Malformed argument objects salvaged from the model's text.
    pub argument_salvages: usize,
    /// Tool-free turns retried for carrying a tool call as prose.
    pub prose_tool_call_retries: usize,
    /// Streams abandoned because no item arrived in time.
    pub stream_timeouts: usize,
}

/// A one-way stop signal for the run in flight.
///
/// A flag plus a wake-up, rather than a channel: a channel that is dropped
/// would read as a stop, and a signal that is never sent cannot be mistaken
/// for one that was.
#[derive(Debug, Default)]
pub struct CancelToken {
    stopped: AtomicBool,
    notify: tokio::sync::Notify,
}

impl CancelToken {
    /// A fresh, unsignalled token.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Ask the run to stop, and wake it if it is waiting.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.notify.notify_one();
    }

    /// Whether a stop has been asked for.
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// Clear the signal, for the next run.
    pub fn reset(&self) {
        self.stopped.store(false, Ordering::SeqCst);
    }

    /// A future that resolves when [`CancelToken::stop`] is called.
    ///
    /// For `select!`: the run polls this alongside the model stream, so a
    /// stop is acted on without waiting for the next token.
    pub fn notified(&self) -> tokio::sync::futures::Notified<'_> {
        self.notify.notified()
    }

    /// [`Error::Cancelled`] if a stop has been asked for.
    ///
    /// The check a tool makes before it commits anything: a stop that arrives
    /// while a command is running must not be followed by the next write.
    pub fn check(&self) -> Result<()> {
        if self.is_stopped() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// One action a run completed.
///
/// Recorded as it happens rather than at the end of the run, so a run that is
/// cancelled or fails still remembers what it already changed. The model's
/// history is only appended to on success, so without this a cancelled run
/// leaves edits on disk that the model does not know it made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionRecord {
    /// A unique id for this action, so two entries can be told apart.
    ///
    /// The model's tool-call id is attached by the orchestration layer, which
    /// is the only place that knows it; until then this is the runtime's own
    /// counter, which is enough to identify an entry.
    pub id: u64,
    /// The run this action belongs to.
    ///
    /// A session outlives its runs, and the journal is the session's. Without
    /// the run id a reader cannot tell what the run in flight has already
    /// done, which is exactly the question a cancelled run has to answer.
    pub run: u64,
    /// The model's tool-call id, when the action came from a tool call.
    ///
    /// `None` for an action the runtime took on its own behalf. It is what
    /// lets a result be attributed to the call that produced it rather than
    /// to whichever result arrived next.
    pub call: Option<String>,
    /// What kind of action it was.
    pub kind: ActionKind,
    /// A one-line description, e.g. `wrote src/main.rs (1.2 kB)`.
    pub summary: String,
}

/// The actions a session has completed, in order.
#[derive(Debug, Default)]
pub struct ActionJournal {
    entries: Mutex<Vec<ActionRecord>>,
}

impl ActionJournal {
    /// An empty journal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed action.
    pub fn record(&self, record: ActionRecord) {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(record);
    }

    /// Everything recorded, in order.
    pub fn entries(&self) -> Vec<ActionRecord> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// How many actions have been recorded.
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget everything, for a new session.
    pub fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    /// Attach the model's call id to the action that call produced.
    ///
    /// The tool cannot do this itself: rig hands a tool its arguments and a
    /// context, but not the id of the call it is answering, so the id is only
    /// knowable to the layer that reads the stream. That layer sees the
    /// result after the action has already been recorded, so the entry is
    /// stamped afterwards rather than written with the id in hand.
    ///
    /// The match is on the run and on the path, and only on an entry that has
    /// no id yet. Two calls in one turn run concurrently, so "the most recent
    /// entry" would be a guess; the path is what the call named, and it is
    /// what makes the stamp land on the right entry.
    ///
    /// Returns whether an entry was stamped.
    pub fn attribute(&self, call: &str, path: &str) -> bool {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let run = entries.last().map(|e| e.run).unwrap_or(0);
        for entry in entries.iter_mut().rev() {
            if entry.run == run && entry.call.is_none() && entry.summary.contains(path) {
                entry.call = Some(call.to_string());
                return true;
            }
        }
        false
    }
}

/// What a command the run actually ran produced.
///
/// The tool knows whether a command succeeded and the session needs to know,
/// but the two are not connected: the rendered output is the *model's* copy of
/// what happened, and a failing `cargo check` is a successful tool call
/// reporting a failed command, so the result's error flag says nothing about
/// whether the check passed. The outcome is therefore recorded here, as it
/// happens, and read back by the orchestration layer.
///
/// A command that was denied never reaches this, which is what keeps a refusal
/// from being counted as a check that failed: nothing ran, so there is nothing
/// to report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutcome {
    /// The command line, as it was run.
    pub command: String,
    /// Whether it exited zero.
    pub passed: bool,
    /// The first line of what it printed, for the transcript.
    pub summary: String,
}

/// What a read produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Read {
    /// The text, up to the read limit.
    pub content: String,
    /// The size on disk, when the file was larger than the read limit.
    ///
    /// `None` means the content is the whole file. The distinction is the
    /// tool's to report: a truncated read that did not say so would look like
    /// a short file.
    pub truncated_from: Option<u64>,
}

/// The next run id.
static NEXT_RUN: AtomicU64 = AtomicU64::new(1);

/// A fresh run id, so a journal entry can be attributed to the run that made
/// it.
pub fn next_run_id() -> u64 {
    NEXT_RUN.fetch_add(1, Ordering::Relaxed)
}

/// The next action id, for the journal.
static NEXT_ACTION: AtomicU64 = AtomicU64::new(1);

/// Everything a tool needs, in one value.
#[derive(Debug)]
pub struct Runtime {
    config: Config,
    workspace: Workspace,
    /// The gate.
    ///
    /// Behind a lock rather than fixed at construction because the binary
    /// builds the runtime before it has a UI to ask on: the interactive
    /// approver needs the channel the event loop reads, and the event loop
    /// needs the app, which needs the runtime. Installing it afterwards
    /// breaks that cycle without leaving a window in which the gate is open:
    /// the mode's default is in place until it is replaced, and for `Prompt`
    /// that default denies.
    approval: RwLock<Arc<dyn Approver>>,
    process: ProcessRunner,
    cancel: Arc<CancelToken>,
    actions: ActionJournal,
    /// What this session last saw in each file it has read.
    ///
    /// A digest rather than the text, so a long session that reads a thousand
    /// files does not hold a thousand files. The entry is what makes "the
    /// model is editing text it has actually seen" checkable: without it, an
    /// edit whose search string happens to still match would be applied to a
    /// file the model has never looked at, and that is a silent wrong edit
    /// rather than a failed one.
    seen: Mutex<HashMap<PathBuf, u64>>,
    /// The size of the most recent request, as the budget hook measured it.
    ///
    /// Written by the hook on every model call and read by the app, which is
    /// what shows it. A lock rather than an atomic because the estimate is a
    /// struct, and the contention is one write per model call.
    estimate: Mutex<Option<Estimate>>,
    /// The run in flight, as the journal attributes actions to it.
    ///
    /// Zero before the first run. The app calls [`Runtime::begin_run`] when it
    /// starts one, so an action recorded by a tool can be attributed to the
    /// run that asked for it rather than to the session as a whole.
    run: AtomicU64,
    /// The outcome of each command that actually ran, in order.
    ///
    /// Drained by the orchestration layer as the results arrive. A queue
    /// rather than a single slot because the calls in one turn run
    /// concurrently, so "the last command" is not a well-defined thing to
    /// read.
    outcomes: Mutex<Vec<CommandOutcome>>,
    /// How often each repair fired, for the eval harness.
    ///
    /// Atomics rather than a mutex: the hooks that write them run inside the
    /// model call, on whatever task the provider drives, and a counter is the
    /// one thing here that is genuinely just a number.
    repairs: RepairCounters,
}

/// The four repair counters, as atomics.
#[derive(Debug, Default)]
struct RepairCounters {
    /// Tool names repaired to a registered tool.
    tool_name_repairs: AtomicUsize,
    /// Malformed argument objects salvaged from the model's text.
    argument_salvages: AtomicUsize,
    /// Tool-free turns retried for carrying a tool call as prose.
    prose_tool_call_retries: AtomicUsize,
    /// Streams abandoned because no item arrived in time.
    stream_timeouts: AtomicUsize,
}

/// What a command's reach adds to its approval prompt.
///
/// Empty when the command reads as staying inside the workspace, which is the
/// common case and must not add a line that says so: a prompt that carried a
/// reassurance on every command would train the eye to skip the one place it
/// matters.
fn reach_note(reach: &Reach) -> String {
    let mut note = String::new();
    if !reach.outside.is_empty() {
        let paths: Vec<String> = reach
            .outside
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        note.push_str(&format!(" · reaches {}", paths.join(", ")));
    }
    if !reach.unresolvable.is_empty() {
        note.push_str(&format!(
            " · uses {}, so what it reaches cannot be read from the command",
            reach.unresolvable.join(", ")
        ));
    }
    note
}

impl Runtime {
    /// A runtime for `workspace`, configured by `config`.
    ///
    /// The approver follows the mode: [`ApprovalMode::Auto`] installs one that
    /// approves everything, and [`ApprovalMode::Prompt`] installs one that
    /// denies everything until an interactive channel is attached with
    /// [`Runtime::with_approver`]. A gate with nobody behind it stays shut,
    /// which is the safe way round: the alternative is a build that silently
    /// writes files because the prompt was never wired up.
    pub fn new(config: Config, workspace: Workspace) -> Arc<Self> {
        let approval: Arc<dyn Approver> = if config.approval.is_auto() {
            Arc::new(AutoApprover)
        } else {
            Arc::new(NoChannel)
        };
        Self::with_approver(config, workspace, approval)
    }

    /// A runtime whose approval decisions come from `approver`.
    pub fn with_approver(
        config: Config,
        workspace: Workspace,
        approver: Arc<dyn Approver>,
    ) -> Arc<Self> {
        let process = ProcessRunner::new(
            workspace.root().to_path_buf(),
            config.command_timeout,
            config.limits.max_shell_bytes,
        );
        Arc::new(Self {
            config,
            workspace,
            approval: RwLock::new(approver),
            process,
            cancel: CancelToken::new(),
            actions: ActionJournal::new(),
            seen: Mutex::new(HashMap::new()),
            estimate: Mutex::new(None),
            run: AtomicU64::new(0),
            outcomes: Mutex::new(Vec::new()),
            repairs: RepairCounters::default(),
        })
    }

    /// Start a run and return its id.
    ///
    /// Every action the tools record from here on is attributed to it, which
    /// is what lets a reader ask what the run in flight has already done. The
    /// id is monotonic and never reused, so an entry from an earlier run
    /// cannot be mistaken for one from this one.
    pub fn begin_run(&self) -> u64 {
        let id = next_run_id();
        self.run.store(id, Ordering::SeqCst);
        id
    }

    /// The run in flight, or zero if none has started.
    pub fn current_run(&self) -> u64 {
        self.run.load(Ordering::SeqCst)
    }

    /// Install `approver` as the gate.
    ///
    /// Called once, by the binary, when the UI that answers the prompt
    /// exists. A runtime that is never given one keeps the mode's default,
    /// which for [`crate::config::ApprovalMode::Prompt`] denies everything.
    pub fn set_approver(&self, approver: Arc<dyn Approver>) {
        *self
            .approval
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = approver;
    }

    /// The configuration in force.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The workspace this session may touch.
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// The process runner.
    pub fn process(&self) -> &ProcessRunner {
        &self.process
    }

    /// The run's stop signal.
    pub fn cancel(&self) -> &Arc<CancelToken> {
        &self.cancel
    }

    /// The record of completed actions.
    pub fn actions(&self) -> &ActionJournal {
        &self.actions
    }

    /// The character budget for one tool result.
    pub fn tool_result_budget(&self) -> usize {
        self.config.context.tool_result_chars
    }

    /// Record what a command that ran produced.
    fn record_outcome(&self, outcome: CommandOutcome) {
        self.outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(outcome);
    }

    /// Take the outcome of the command `command`, if it ran.
    ///
    /// Removes the entry, so a command run twice in one turn is reported twice
    /// rather than the first outcome being read for both. `None` means the
    /// command never ran: it was denied, or it never got as far as the runner.
    pub fn take_outcome(&self, command: &str) -> Option<CommandOutcome> {
        let mut outcomes = self
            .outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let at = outcomes.iter().position(|o| o.command == command)?;
        Some(outcomes.remove(at))
    }

    /// Record that a tool name was repaired to a registered tool.
    pub fn record_tool_name_repair(&self) {
        self.repairs
            .tool_name_repairs
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a malformed argument object was salvaged from the model's
    /// text.
    pub fn record_argument_salvage(&self) {
        self.repairs
            .argument_salvages
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a tool-free turn was retried for carrying a tool call as
    /// prose.
    pub fn record_prose_tool_call_retry(&self) {
        self.repairs
            .prose_tool_call_retries
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a stream was abandoned because no item arrived in time.
    pub fn record_stream_timeout(&self) {
        self.repairs.stream_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    /// How often each repair has fired in this session.
    ///
    /// Read once, at the end of a run. The counters are never reset: they
    /// describe the session, not the run, which is what makes them comparable
    /// across the tasks of one eval sweep.
    pub fn repair_report(&self) -> RepairReport {
        RepairReport {
            tool_name_repairs: self.repairs.tool_name_repairs.load(Ordering::Relaxed),
            argument_salvages: self.repairs.argument_salvages.load(Ordering::Relaxed),
            prose_tool_call_retries: self.repairs.prose_tool_call_retries.load(Ordering::Relaxed),
            stream_timeouts: self.repairs.stream_timeouts.load(Ordering::Relaxed),
        }
    }

    /// Record the size of the request the budget hook just measured.
    ///
    /// The hook runs inside the model call, where the app cannot see it, and
    /// the app is what shows the figure. Recording it here is what keeps the
    /// estimate and the provider's own usage report from being two different
    /// accountings of the same request: the hook measures, the app displays.
    pub fn record_estimate(&self, estimate: Estimate) {
        *self
            .estimate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(estimate);
    }

    /// The size of the most recent request, as the budget hook measured it.
    ///
    /// `None` until a model call has been made. This is an estimate of a
    /// request that has not been sent yet, and it is deliberately separate
    /// from the provider's usage report, which is exact and describes a call
    /// that has already happened.
    pub fn last_estimate(&self) -> Option<Estimate> {
        *self
            .estimate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Ask whether `request` may run.
    ///
    /// The single gate. Every mutating tool goes through it, so a denial is
    /// turned into a result in exactly one place and no tool can forget to
    /// ask. A stop that has already been asked for denies without consulting
    /// the approver: there is no point prompting for an action the run is
    /// about to abandon.
    pub async fn approve(&self, request: ApprovalRequest) -> Result<()> {
        self.cancel.check()?;
        let approver = self
            .approval
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        match approver.decide(request).await {
            Decision::Approved => Ok(()),
            Decision::Denied(reason) => Err(Error::Denied(reason)),
        }
    }

    /// Resolve `rel`, asking the user when it lands outside the workspace.
    ///
    /// The workspace is the one directory a session owns, and a path outside
    /// it used to be refused outright. That is the wrong answer to a normal
    /// request: a reference crate beside the project, or a config in the
    /// user's home, is a thing the work genuinely needs, and a refusal only
    /// moves the problem to the user's hands. So the refusal became a
    /// question, asked about the *directory* rather than the path, and
    /// answered once: the grant is remembered for the rest of the session
    /// ([`Workspace::grant`]), so the next file in the same directory does not
    /// ask again, and nothing is written to disk, so the next session starts
    /// with the root alone.
    ///
    /// The question is asked only when the path is outside. A path inside the
    /// root resolves exactly as it always did, which is what keeps the common
    /// case free of prompts.
    ///
    /// `resolve` is passed in rather than chosen here because the two rules
    /// differ in what they follow: a path that must exist is canonicalised
    /// whole, and a path about to be created is canonicalised up to its
    /// parent. Both are re-run after a grant, so what the caller gets back is
    /// the path that rule produces and not one this method assembled.
    async fn resolve_asking(
        &self,
        rel: &str,
        resolve: impl Fn(&Workspace, &str) -> Result<PathBuf>,
    ) -> Result<PathBuf> {
        match resolve(&self.workspace, rel) {
            Err(Error::OutsideWorkspace { .. }) => {}
            settled => return settled,
        }

        // The real path, which is the only form the user can judge:
        // `../../etc/passwd`, `/etc/passwd` and a link to either are the same
        // question. A path whose shape cannot be checked fails here rather
        // than being asked about, because there is nothing to ask.
        let target = self.workspace.real_target_of(rel)?;
        self.ask_access(rel, &target, &format!("'{rel}' is outside the workspace."))
            .await?;

        // Re-resolved rather than returned from above: the grant changed what
        // the rule accepts, and the rule is what decides, including the
        // symlink check that a lexically fine path can still fail.
        resolve(&self.workspace, rel)
    }

    /// Put one directory outside the workspace to the user, and record the
    /// answer.
    ///
    /// `shown` is what the action named — a tool's path argument, or the
    /// command line — and `why` is the first line of the prompt, because the
    /// two callers are asking about the same directory for different reasons
    /// and the reason is what the user is judging.
    async fn ask_access(&self, shown: &str, target: &Path, why: &str) -> Result<()> {
        // The grant is a directory. For a file that is its parent; for a
        // directory it is the directory itself, so pointing `list_dir` at a
        // sibling project does not quietly grant everything beside it.
        let dir = match tokio::fs::metadata(target).await {
            Ok(meta) if meta.is_dir() => target.to_path_buf(),
            _ => target.parent().unwrap_or(target).to_path_buf(),
        };

        let request = ApprovalRequest::new(ActionKind::Access, format!("access {}", dir.display()))
            .with_path(shown.to_string())
            .with_preview(format!(
                "{why}\n\nworkspace: {}\nrequested: {}",
                self.workspace.root().display(),
                target.display()
            ))
            .with_detail(format!(
                "approving grants {} for the rest of this session",
                dir.display()
            ));
        self.approve(request).await?;
        self.workspace.grant(dir);
        Ok(())
    }

    /// Gate what `command` appears to reach before it is put to the user as a
    /// command.
    ///
    /// The shell is not confined: an approved command runs with the root as
    /// its working directory and is otherwise free to name any path it likes.
    /// What this does is stop the gate running blind. The paths in the
    /// command are resolved ([`Workspace::reach_of`]), and each one outside
    /// the workspace is asked about as a directory — the same question
    /// `read_file` would have to ask, so the shell is not a way around the
    /// boundary the file tools enforce.
    ///
    /// What cannot be resolved from a string is returned rather than refused:
    /// a variable or a substitution is ordinary in a real command line, and
    /// the honest answer is to name it in the prompt so the person answering
    /// knows the resolved paths are not the whole story. The one place that
    /// is not enough is auto mode, where nobody reads the prompt, and there
    /// such a command is refused: a mode that cannot ask cannot be given a
    /// command whose reach cannot be read.
    ///
    /// Returns the reach, so the caller can show it in the prompt it is about
    /// to put up.
    async fn gate_reach(&self, command: &str) -> Result<Reach> {
        let reach = self.workspace.reach_of(command);
        if !reach.unresolvable.is_empty() && self.config.approval.is_auto() {
            return Err(Error::Denied(format!(
                "'{command}' uses {}, so what it reaches cannot be read from it, and \
                 auto-approval cannot ask; write the paths out, or run it without \
                 auto-approval",
                reach.unresolvable.join(", ")
            )));
        }
        for target in &reach.outside {
            // A grant just given may already cover this one: the user
            // answered for a directory, not for a path.
            if self.workspace.may_touch(target) {
                continue;
            }
            self.ask_access(
                command,
                target,
                &format!(
                    "the command reaches '{}', which is outside the workspace.",
                    target.display()
                ),
            )
            .await?;
        }
        Ok(reach)
    }

    /// Resolve a path that must exist, asking if it is outside the workspace.
    ///
    /// This is what the read-only tools (`list_dir`, `grep`, `glob`) resolve
    /// with. They reach the filesystem through `search`, not through this
    /// service, so the question has to be asked here or not at all.
    pub async fn resolve_existing(&self, rel: &str) -> Result<PathBuf> {
        self.resolve_asking(rel, |ws, rel| ws.resolve_existing(rel))
            .await
    }

    /// Resolve a path about to be created or replaced, asking if it is
    /// outside the workspace.
    pub async fn resolve_for_write(&self, rel: &str) -> Result<PathBuf> {
        self.resolve_asking(rel, |ws, rel| ws.resolve_for_write(rel))
            .await
    }

    // ------------------------------------------------- the execution service
    //
    // The gate is enforced *here*, not in the tools. A tool that called
    // `approve` and then did the work itself would be one refactor away from
    // doing the work without asking; a tool that can only reach the
    // filesystem through these methods cannot. That is also what makes the
    // examples and a direct tool call in a test go through the same gate as
    // the binary: there is no other way in.

    /// Create or replace `rel` with `content`, after asking.
    ///
    /// `rel` is the workspace-relative path the user is shown *and* the path
    /// that is resolved. There is no separate resolved-path argument on
    /// purpose: a caller that could pass one could pass one that does not
    /// match what the user was shown, and the approval would then be for a
    /// different file than the write.
    pub async fn write_file(&self, rel: &str, content: &str) -> Result<()> {
        let path = self.resolve_for_write(rel).await?;
        let version = self.version_of(&path).await?;
        let (kind, preview) = match version.text() {
            Some(old) => (ActionKind::Write, diff_preview(old, content)),
            None => (ActionKind::Create, new_file_preview(content)),
        };
        let request = ApprovalRequest::new(kind, format!("{} {rel}", kind.label()))
            .with_path(rel)
            .with_preview(preview)
            .with_detail(format!("{} bytes", content.len()));
        self.approve(request).await?;
        self.commit(kind, rel, &version, content).await
    }

    /// Read `rel`, remembering what it held.
    ///
    /// The read goes through the service rather than the tool because the
    /// remembering is the point: an edit is checked against what this session
    /// last saw, and a read that did not record would leave the check with
    /// nothing to compare against.
    ///
    /// The content is returned whole up to the read limit. Eliding it to the
    /// model's budget is presentation and stays in the tool; the digest is
    /// taken over the whole file either way, so a truncated view does not make
    /// the next edit look stale.
    pub async fn read_file(&self, rel: &str) -> Result<Read> {
        // `resolve_existing` rather than `resolve`: the lexical check alone
        // would accept `link/secret.txt` where `link` points out of the
        // workspace. This one canonicalises the real target and re-tests it.
        let path = self.resolve_existing(rel).await?;
        let meta = tokio::fs::metadata(&path)
            .await
            .map_err(|e| Error::io("reading", &path, e))?;
        // A directory reads as an opaque OS error ("access is denied" on
        // Windows), which tells the model nothing about what to do instead.
        if meta.is_dir() {
            return Err(Error::invalid(format!(
                "'{rel}' is a directory; use list_dir to list it"
            )));
        }
        // A device, a FIFO or a socket is not a file: reading one can block
        // forever, and the tool has no deadline of its own.
        if !meta.is_file() {
            return Err(Error::invalid(format!("'{rel}' is not a regular file")));
        }

        let limit = self.config.limits.max_read_bytes;
        let over = meta.len() > limit;
        let content = if over {
            read_head(&path, limit as usize).await
        } else {
            tokio::fs::read_to_string(&path).await
        }
        .map_err(|e| Error::io("reading", &path, e))?;

        self.note_seen(&path, &Version::Text(content.clone()));
        Ok(Read {
            content,
            truncated_from: over.then_some(meta.len()),
        })
    }

    /// Apply `blocks` to `rel`, after asking.
    ///
    /// The read, the check and the write are one operation on purpose. Split
    /// across a tool body they would be three steps with a gap between the
    /// second and the third, and the version check is only meaningful if the
    /// bytes it compares are the bytes the replacement was computed from.
    ///
    /// Every block is applied to an in-memory copy before anything is
    /// committed, so a patch that fails on its third block leaves the file
    /// exactly as it was. The diff the user is shown is the diff of the whole
    /// patch, not of the first block.
    pub async fn edit_file(&self, rel: &str, blocks: &[Block]) -> Result<()> {
        let path = self.resolve_existing(rel).await?;
        // The whole file is read to compute the replacement, so a file too
        // large to read is a file too large to edit. Refusing here rather
        // than letting the read fail gives the model something it can act on.
        //
        // A missing file is answered with the tool that *can* create one.
        // "The system cannot find the file specified" is true and useless:
        // the model's next move is `write_file`, and the error should say so.
        let meta = match tokio::fs::metadata(&path).await {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::invalid(format!(
                    "'{rel}' does not exist; use write_file to create it"
                )));
            }
            Err(e) => return Err(Error::io("reading", &path, e)),
        };
        if !meta.is_file() {
            return Err(Error::invalid(format!("'{rel}' is not a regular file")));
        }
        let limit = self.config.limits.max_read_bytes;
        if meta.len() > limit {
            return Err(Error::invalid(format!(
                "'{rel}' is {} bytes, over the {limit}-byte limit for an edit; \
                 use write_file to replace it whole",
                meta.len()
            )));
        }
        let version = self.version_of(&path).await?;
        // The file was there a moment ago and is not now. Rare, but the
        // alternative is treating "it vanished" as "it is empty" and writing
        // a patch computed against nothing.
        let Some(old) = version.text() else {
            return Err(Error::invalid(format!(
                "'{rel}' does not exist; use write_file to create it"
            )));
        };

        // The file must still be what this session read. The search string
        // matching is not enough: it can match text the model has never seen,
        // and the edit would then land in a file that has moved on.
        if let Some(seen) = self.seen(&path) {
            if seen != version.fingerprint() {
                return Err(Error::StaleRead {
                    path: rel.to_string(),
                });
            }
        }

        let new_content = patch::apply(old, blocks)?;
        let request = ApprovalRequest::new(ActionKind::Patch, format!("patch {rel}"))
            .with_path(rel)
            .with_preview(diff_preview(old, &new_content))
            .with_detail(format!(
                "{} edit{} · {} bytes -> {} bytes",
                blocks.len(),
                if blocks.len() == 1 { "" } else { "s" },
                old.len(),
                new_content.len()
            ));
        self.approve(request).await?;
        self.commit(ActionKind::Patch, rel, &version, &new_content)
            .await
    }

    /// Run `command` in the workspace, after asking.
    ///
    /// The prompt carries the command, the directory it runs in, the deadline
    /// and what the command's text says it reaches, because those are what
    /// decide what it can touch and how long it can hold the session. A path
    /// in the command that lands outside the workspace is granted first, by
    /// [`Runtime::gate_reach`], so the shell is not a way around the boundary
    /// the file tools enforce.
    ///
    /// The command is watched by the run's cancel token, so a stop kills it
    /// and everything it spawned rather than waiting for the deadline. The
    /// output it produced before the kill is kept: a cancelled build has
    /// usually already printed the error that made the user stop it.
    pub async fn run_command(&self, command: &str) -> Result<CommandOutput> {
        let reach = self.gate_reach(command).await?;
        let request = ApprovalRequest::new(ActionKind::Command, format!("run {command}"))
            .with_preview(command.to_string())
            .with_detail(format!(
                "in {} · timeout {}s{}",
                self.workspace.root().display(),
                self.config.command_timeout.as_secs(),
                reach_note(&reach)
            ));
        self.approve(request).await?;
        let out = self.process.run_shell_until(command, &self.cancel).await?;
        self.record_outcome(CommandOutcome {
            command: command.to_string(),
            passed: out.succeeded(),
            summary: out.first_line(),
        });
        self.record(
            ActionKind::Command,
            format!(
                "ran {command} ({}, {:.1}s)",
                out.termination.describe(),
                out.duration.as_secs_f64()
            ),
        );
        Ok(out)
    }

    /// Run `program` with `args`, after asking.
    ///
    /// The structured counterpart of [`Runtime::run_command`]: no shell in
    /// between, so a path containing a space, a quote or a `&` is one argument
    /// rather than several, and nothing in the argument is interpreted. Used
    /// for the commands the agent runs on its own behalf, where the program is
    /// known and only the arguments vary.
    ///
    /// It goes through the same gate as the shell form. A structured command is
    /// easier to read, not safer: `cargo test` runs whatever the test binary
    /// does, so it is still an action the user is asked about.
    pub async fn run_program(&self, program: &str, args: &[&str]) -> Result<CommandOutput> {
        let line = crate::process::program_line(program, args);
        let reach = self.gate_reach(&line).await?;
        let request = ApprovalRequest::new(ActionKind::Command, format!("run {line}"))
            .with_preview(line.clone())
            .with_detail(format!(
                "in {} · timeout {}s{}",
                self.workspace.root().display(),
                self.config.command_timeout.as_secs(),
                reach_note(&reach)
            ));
        self.approve(request).await?;
        let out = self
            .process
            .run_program_until(program, args, &self.cancel)
            .await?;
        self.record(
            ActionKind::Command,
            format!(
                "ran {line} ({}, {:.1}s)",
                out.termination.describe(),
                out.duration.as_secs_f64()
            ),
        );
        Ok(out)
    }

    /// The version of `path`, as an [`Error`] rather than an `io::Error`.
    async fn version_of(&self, path: &Path) -> Result<Version> {
        Version::of(path)
            .await
            .map_err(|e| Error::io("reading", path, e))
    }

    /// The digest this session last saw for `path`, if it has read it.
    fn seen(&self, path: &Path) -> Option<u64> {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(path)
            .copied()
    }

    /// Remember what `path` holds now.
    ///
    /// Called after a read and after a successful write, so the entry always
    /// describes the file as this session last left it. A write that did not
    /// update it would make the session's own change look like someone else's.
    fn note_seen(&self, path: &Path, version: &Version) {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(path.to_path_buf(), version.fingerprint());
    }

    /// Write `content` over `rel`, refusing if the file moved on.
    ///
    /// The version check is the second half of the approval: the user agreed
    /// to a diff against specific bytes, so a file that is no longer those
    /// bytes is a different change and needs a new preview. The check is
    /// repeated after the approval rather than only before it, because the
    /// user may have taken a while to answer, and that is exactly the window
    /// in which an editor would have saved.
    ///
    /// The path is resolved **again** here, for the same reason. The first
    /// resolution happened before the prompt; a directory in the chain could
    /// have been replaced by a link while the user was reading the diff, and
    /// the second resolution is what notices. It is not a complete defence
    /// against a swap between this line and the rename, but the sibling temp
    /// file makes that harmless: see [`filesystem::write_atomic`].
    async fn commit(
        &self,
        kind: ActionKind,
        rel: &str,
        expected: &Version,
        content: &str,
    ) -> Result<()> {
        self.cancel.check()?;
        let path = self.workspace.resolve_for_write(rel)?;
        let now = self.version_of(&path).await?;
        if !now.is(expected) {
            return Err(Error::Stale {
                path: rel.to_string(),
            });
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| Error::io("creating", parent, e))?;
            }
        }
        filesystem::write_atomic(&path, content)
            .await
            .map_err(|e| Error::io("writing", &path, e))?;
        self.note_seen(&path, &Version::Text(content.to_string()));
        self.record(kind, format!("wrote {rel} ({} bytes)", content.len()));
        Ok(())
    }

    /// Add an entry to the journal.
    ///
    /// The run id is read here rather than passed in, so a tool cannot record
    /// an action against a run that is not the one that asked for it. The
    /// call id is left to the orchestration layer, which is the only place
    /// that knows which call a result belongs to.
    fn record(&self, kind: ActionKind, summary: String) {
        self.actions.record(ActionRecord {
            id: NEXT_ACTION.fetch_add(1, Ordering::Relaxed),
            run: self.current_run(),
            call: None,
            kind,
            summary,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::ScriptedApprover;
    use crate::config::ApprovalMode;

    fn workspace() -> Workspace {
        Workspace::new(std::env::temp_dir()).expect("the temp dir is a valid root")
    }

    #[test]
    fn a_prompt_mode_runtime_denies_until_a_channel_is_attached() {
        let rt = Runtime::new(Config::defaults(), workspace());
        let decision = futures::executor::block_on(
            rt.approve(ApprovalRequest::new(ActionKind::Write, "write a")),
        );
        assert!(matches!(decision, Err(Error::Denied(_))), "{decision:?}");
    }

    #[test]
    fn an_auto_mode_runtime_approves() {
        let config = Config::defaults().with_approval(ApprovalMode::Auto);
        let rt = Runtime::new(config, workspace());
        let decision = futures::executor::block_on(
            rt.approve(ApprovalRequest::new(ActionKind::Write, "write a")),
        );
        assert!(decision.is_ok(), "{decision:?}");
    }

    #[test]
    fn a_stopped_run_denies_without_asking() {
        let approver = Arc::new(ScriptedApprover::approving());
        let rt = Runtime::with_approver(Config::defaults(), workspace(), approver.clone());
        rt.cancel().stop();
        let decision = futures::executor::block_on(
            rt.approve(ApprovalRequest::new(ActionKind::Write, "write a")),
        );
        assert!(matches!(decision, Err(Error::Cancelled)), "{decision:?}");
        assert_eq!(
            approver.count(),
            0,
            "a stopped run must not prompt for an action it is abandoning"
        );
    }

    #[test]
    fn the_process_runner_starts_in_the_workspace() {
        let rt = Runtime::new(Config::defaults(), workspace());
        assert_eq!(rt.process().cwd(), rt.workspace().root());
        assert_eq!(rt.process().timeout(), rt.config().command_timeout);
    }

    #[test]
    fn the_journal_records_in_order() {
        let journal = ActionJournal::new();
        assert!(journal.is_empty());
        journal.record(ActionRecord {
            id: 1,
            run: 7,
            call: Some("call_1".to_string()),
            kind: ActionKind::Write,
            summary: "wrote a".to_string(),
        });
        journal.record(ActionRecord {
            id: 2,
            run: 7,
            call: None,
            kind: ActionKind::Command,
            summary: "ran cargo check".to_string(),
        });
        assert_eq!(journal.len(), 2);
        assert_eq!(journal.entries()[0].summary, "wrote a");
        assert_eq!(journal.entries()[0].run, 7);
        assert_eq!(journal.entries()[0].call.as_deref(), Some("call_1"));
        journal.clear();
        assert!(journal.is_empty());
    }

    #[test]
    fn a_run_id_is_monotonic_and_never_reused() {
        let rt = Runtime::new(Config::defaults(), workspace());
        assert_eq!(rt.current_run(), 0, "no run has started yet");
        let first = rt.begin_run();
        assert_eq!(rt.current_run(), first);
        let second = rt.begin_run();
        assert!(second > first, "a run id is never reused");
        assert_eq!(rt.current_run(), second);
    }

    #[test]
    fn a_cancel_token_is_one_way_until_reset() {
        let token = CancelToken::new();
        assert!(!token.is_stopped());
        assert!(token.check().is_ok());
        token.stop();
        assert!(token.is_stopped());
        assert!(matches!(token.check(), Err(Error::Cancelled)));
        token.reset();
        assert!(token.check().is_ok());
    }
}

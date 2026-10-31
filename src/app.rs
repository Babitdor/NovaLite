use futures::{FutureExt, StreamExt};
use rig::agent::{Agent, MultiTurnStreamItem};
use rig::message::Message as RigMessage;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use crate::event::Event;
use crate::runtime::CancelToken;

/// Most content rows the input box will show before it scrolls.
///
/// A pasted file is a legitimate prompt, but it must not push the
/// conversation off the screen; past this the box scrolls to the caret.
pub const MAX_INPUT_ROWS: u16 = 8;

/// How many tool results keep their full text in memory.
///
/// The transcript is a *view* of the conversation and is bounded by
/// [`crate::render::MAX_TRANSCRIPT_LINES`], but the conversation itself is
/// not: a long session accumulates every result it ever read, and one result
/// can be as large as the context budget allows. The rows stay, because the
/// transcript is the user's record of what happened and dropping a row would
/// rewrite history; the bodies are released, because a body that has scrolled
/// far out of the view is only ever read again by scrolling back and
/// expanding the row, and the row still says what the call was and how it
/// ended.
pub const MAX_RETAINED_BODIES: usize = 200;

/// What a released body is replaced with.
///
/// It has to read as a deliberate act rather than as an empty result: a row
/// that says nothing looks like a call that returned nothing, which is a
/// different and more alarming thing.
pub const RELEASED_BODY: &str = "(released to keep the session's memory bounded)";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    /// A tool call, rendered in the transcript as a collapsible block.
    Tool,
}

/// How a tool call ended.
///
/// The transcript's marker is the whole of what a user reads at a glance, so
/// the states it can be in have to be the states that actually differ. A
/// refusal is not a failure: the tool worked, the user said no, and painting
/// it with the same cross as a broken build tells the user their agent is
/// broken when it is doing what they told it to. A cancellation is not a
/// failure either: nothing was wrong with the call, the run was abandoned
/// while it was in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolOutcome {
    /// The call is still running.
    #[default]
    Pending,
    /// The call returned a result.
    Ok,
    /// The call failed.
    Failed,
    /// The user refused the action.
    Denied,
    /// The run was cancelled while the call was in flight.
    Cancelled,
}

impl ToolOutcome {
    /// Whether the call ended without doing what it was asked to.
    ///
    /// A denial and a cancellation are not failures, but they are not
    /// successes either, and a group summary that counted them as either
    /// would be wrong about what the run did.
    pub fn is_failure(self) -> bool {
        matches!(self, Self::Failed)
    }

    /// The word the transcript uses for this outcome.
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "running",
            Self::Ok => "ok",
            Self::Failed => "failed",
            Self::Denied => "denied",
            Self::Cancelled => "cancelled",
        }
    }
}

/// A tool call as it appears in the conversation.
///
/// The block is created when the model opens the call's part, so it is visible
/// while the arguments are still streaming, and is filled in as the call
/// commits and its result comes back. It is collapsed by default: a run with
/// several calls would otherwise bury the answer under raw JSON.
#[derive(Debug, Clone, Default)]
pub struct ToolBlock {
    pub name: String,
    /// Argument summary, or the raw JSON while it is still streaming.
    pub args: String,
    /// Raw argument JSON received so far while the call streams.
    ///
    /// Each delta is a fragment, not a document, so the display is rebuilt
    /// from the whole buffer rather than from the fragment alone. It is
    /// cleared when the call commits and the validated arguments replace it.
    pub args_json: String,
    /// The result body, empty until the call returns.
    pub result: String,
    /// How the call ended.
    pub outcome: ToolOutcome,
    /// True when the result was cut to fit the context budget.
    ///
    /// The model was told, in the result itself, that content was dropped. The
    /// user is told too: a result that reads as complete when it is not is the
    /// one way a truncated answer can mislead.
    pub truncated: bool,
    /// True while the arguments are still arriving.
    pub streaming: bool,
    /// True once the result has been recorded.
    pub done: bool,
    /// The part this block was opened for, while it is still a placeholder.
    ///
    /// A call has no id until its part ends, so the part's position is the
    /// only identity the streaming placeholder has. It is what
    /// [`Event::ToolCallIdentified`] matches on.
    pub part: Option<usize>,
    /// The provider's id for the call, once it is known.
    ///
    /// This is the identity the execution commit and the result are keyed by,
    /// so it is what the block is found by once the call has committed.
    pub call: Option<String>,
    /// True once the result body has been released to bound memory.
    ///
    /// The row stays and still says what the call was and how it ended; only
    /// the text is gone. See [`MAX_RETAINED_BODIES`].
    pub released: bool,
}

impl ToolBlock {
    /// Whether the call failed.
    ///
    /// A denial and a cancellation are deliberately not failures: the group
    /// summary counts what went wrong, and a user who declined an action has
    /// not seen anything go wrong.
    pub fn is_error(&self) -> bool {
        self.outcome.is_failure()
    }

    /// One-line summary shown when the block is collapsed.
    pub fn summary(&self) -> String {
        let mut parts = vec![self.name.clone()];
        if !self.args.is_empty() {
            parts.push(self.args.clone());
        }
        if self.streaming {
            parts.push("streaming…".to_string());
        } else if self.done {
            let lines = self.result.lines().count().max(1);
            let mut tail = format!("{lines} line{}", if lines == 1 { "" } else { "s" });
            if self.truncated {
                tail.push_str(" (truncated)");
            }
            if self.outcome != ToolOutcome::Ok {
                tail.push_str(" · ");
                tail.push_str(self.outcome.label());
            }
            parts.push(tail);
        }
        parts.join("  ·  ")
    }
}

/// A run of consecutive tool calls of one category, as one block in the
/// transcript.
///
/// The transcript keeps every call, but a run of eight reads is one thing the
/// agent did, not eight. Grouping them under a heading is what makes a long
/// exploration legible without hiding any of it.
#[derive(Debug, Clone)]
pub struct ToolGroup {
    /// Index of the group's first message. It is the group's identity: the
    /// fold state is keyed by it, and it is stable for the life of the
    /// conversation because messages are only ever appended.
    pub first: usize,
    /// Index of the group's last message, so the transcript can step over the
    /// whole group at once.
    pub last: usize,
    pub category: crate::tools::ToolCategory,
    /// The group's message indices, in order.
    pub members: Vec<usize>,
}

impl ToolGroup {
    /// How many calls the group holds.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the group holds no calls.
    ///
    /// A group is built from at least one call, so this is false for every
    /// group the transcript can produce. It exists because a public `len`
    /// without it is a trap for a caller that reaches for the pair.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Total lines of result the group's calls returned, counting a call that
    /// has not returned yet as one line so the number never goes backwards.
    pub fn result_lines(&self, messages: &[ChatMessage]) -> usize {
        self.members
            .iter()
            .filter_map(|i| messages.get(*i))
            .filter_map(|m| m.tool.as_ref())
            .map(|b| b.result.lines().count().max(1))
            .sum()
    }

    /// How many of the group's calls failed.
    pub fn errors(&self, messages: &[ChatMessage]) -> usize {
        self.members
            .iter()
            .filter_map(|i| messages.get(*i))
            .filter_map(|m| m.tool.as_ref())
            .filter(|b| b.is_error())
            .count()
    }

    /// The group's calls by tool name, most frequent first, as
    /// `read_file ×6 · list_dir ×2`. Shown when the group is folded, so a
    /// folded group still says what it did.
    pub fn breakdown(&self, messages: &[ChatMessage]) -> String {
        let mut counts: Vec<(String, usize)> = Vec::new();
        for i in &self.members {
            let Some(name) = messages
                .get(*i)
                .and_then(|m| m.tool.as_ref())
                .map(|b| b.name.clone())
            else {
                continue;
            };
            match counts.iter_mut().find(|(n, _)| *n == name) {
                Some((_, c)) => *c += 1,
                None => counts.push((name, 1)),
            }
        }
        // Most frequent first, then alphabetically, so the order is stable
        // for equal counts rather than dependent on arrival order.
        counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        counts
            .into_iter()
            .map(|(name, n)| {
                if n == 1 {
                    name
                } else {
                    format!("{name} ×{n}")
                }
            })
            .collect::<Vec<_>>()
            .join(" · ")
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// Set only for [`Role::Tool`].
    pub tool: Option<ToolBlock>,
    /// When the message was added, so the transcript can show its age. A
    /// monotonic instant rather than a wall clock: the age is all that is
    /// shown, and it cannot go backwards when the system clock is adjusted.
    pub created: std::time::Instant,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool: None,
            created: std::time::Instant::now(),
        }
    }

    pub fn tool(block: ToolBlock) -> Self {
        Self {
            role: Role::Tool,
            content: String::new(),
            tool: Some(block),
            created: std::time::Instant::now(),
        }
    }
}

/// What an activity row records. The activity is the run's timeline: tool
/// calls, their execution, their results, and per-call accounting. Nothing
/// paints it since the layout became one column, but it is kept: the evals and
/// the live examples read the timeline directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityKind {
    ToolCall,
    ToolExecuted,
    ToolResult,
    Completion,
    Retry,
    /// The run moved to a new stage of the loop.
    Stage,
}

#[derive(Debug, Clone)]
pub struct Activity {
    pub kind: ActivityKind,
    pub title: String,
    pub detail: String,
    pub is_error: bool,
    /// The part a live tool-call row was opened for.
    ///
    /// Part positions restart at zero on every model call, so this is only
    /// unique among the placeholders of the turn being streamed. It is what
    /// keeps two calls that are named at the same time from sharing one row.
    pub part: Option<usize>,
}

/// The `/model` picker: the models Ollama reports, and which one is
/// highlighted.
#[derive(Debug, Clone, Default)]
pub struct ModelPicker {
    pub models: Vec<String>,
    pub selected: usize,
    /// True while `ollama list` is still running.
    pub loading: bool,
    /// Why the list could not be fetched, if it could not.
    pub error: Option<String>,
}

impl ModelPicker {
    /// Move the highlight by `delta`, clamped to the list.
    pub fn move_by(&mut self, delta: i32) {
        if self.models.is_empty() {
            return;
        }
        let last = self.models.len() - 1;
        let next = self.selected as i32 + delta;
        self.selected = next.clamp(0, last as i32) as usize;
    }

    /// The highlighted model, if there is one.
    pub fn current(&self) -> Option<&str> {
        self.models.get(self.selected).map(|s| s.as_str())
    }
}

pub struct App {
    pub messages: Vec<ChatMessage>,
    /// The prompt being composed.
    ///
    /// Private, with [`App::input`] and [`App::set_input`] as the only way in
    /// and out: the caret is a second piece of state that has to agree with
    /// it, and a public field would let a caller replace the buffer and leave
    /// the caret pointing into the middle of the new text.
    input: String,
    /// Caret position in `input`, as a **character** index rather than a byte
    /// offset: the buffer holds whatever the user pastes, and slicing a
    /// multi-byte glyph at a byte offset panics.
    ///
    /// It is clamped on every use rather than kept in step by hand, so it can
    /// never point past the end of the buffer.
    cursor: usize,
    pub status: String,
    pub model_name: String,
    /// The workspace root's directory name, which is what the user calls this
    /// project.
    ///
    /// Set from the session's workspace rather than read from the process's
    /// working directory: the root is what the tools are confined to, and it
    /// is the directory named on the command line when one was given. The
    /// default is the process's own directory name, which is what the root is
    /// when no directory was named.
    pub project_name: String,
    pub should_quit: bool,
    /// Prompts the user has sent, oldest first, for Up/Down recall.
    history_input: Vec<String>,
    /// Where the user is in [`App::history_input`] while browsing it. `None`
    /// means "not browsing": the buffer holds a fresh prompt.
    history_pos: Option<usize>,
    /// The prompt being composed when history browsing started, so Down past
    /// the newest entry restores it instead of losing it.
    history_draft: String,
    /// Cancels the run in flight.
    cancel: Option<Arc<CancelToken>>,
    /// True between the user asking to stop and the run confirming it has.
    pub cancelling: bool,
    /// Prompts typed while a run was in flight, sent in order as each run
    /// settles. Two runs at once would interleave into one transcript and
    /// fight over the assistant message, so a second prompt waits its turn
    /// rather than being refused (which would lose what the user typed).
    pending: std::collections::VecDeque<String>,
    /// Timeline of tool activity and per-call accounting for the current run.
    pub activity: Vec<Activity>,
    /// Message bodies as they were last rendered, kept between frames.
    ///
    /// Parsing markdown is the expensive part of drawing an answer, and the
    /// transcript is redrawn on every tick: ten times a second while a run is
    /// in flight. Re-parsing every answer in a long conversation ten times a
    /// second produces the same lines every time, and it is why a long session
    /// gets slower to watch than a short one. See [`crate::render::BodyCache`].
    pub(crate) body_cache: std::cell::RefCell<crate::render::BodyCache>,
    /// Live reasoning text of the turn in flight.
    pub reasoning: String,
    pub usage_in: u64,
    pub usage_out: u64,
    pub calls: usize,
    /// The prompt length of the most recent model call, in tokens: the size of
    /// the context the model was actually sent.
    ///
    /// This is not `usage_in`, which accumulates across every call of a run
    /// and so grows without bound. The context size is the *last* call's
    /// input tokens, which is what the compaction threshold is measured
    /// against. Zero until a run reports usage.
    pub context_tokens: u64,
    /// The size of the most recent request, in tokens, as the budget hook
    /// measured it before sending it.
    ///
    /// Deliberately separate from [`App::context_tokens`]. That one is the
    /// provider's own report of a call that has already happened, and it is
    /// exact; this one is an estimate of a request that has not been sent, and
    /// it is labelled as one wherever it is shown. They answer different
    /// questions: "how full was the window" and "will the next request fit".
    pub estimated_tokens: u64,
    /// Whether the prompt in flight has already spent its one recovery
    /// attempt.
    ///
    /// A request the provider rejects for its size is retried once against a
    /// compacted history, and only once: a second rejection means compaction
    /// is not what is wrong, and retrying would loop. Reset when the user
    /// sends a new prompt, so the next one gets its own attempt.
    recovery_used: bool,
    /// The prompt to re-send once the recovery compaction lands.
    ///
    /// Held rather than pushed back onto the queue because the queue is the
    /// user's, in the order they typed it: a retry that jumped it would run
    /// before a prompt that was typed first.
    retry_after_compaction: Option<String>,
    /// True while a compaction run is in flight.
    pub compacting: bool,
    /// First transcript line shown, and whether the view follows the tail.
    ///
    /// `auto_scroll` is the user's intent, not the app's. It is armed by
    /// sending a prompt and by `Ctrl+End`, and disarmed by scrolling. Nothing
    /// that happens to a run may set it: a run that finishes while the user is
    /// reading an earlier answer would otherwise pull the viewport out from
    /// under them, which is the one moment they are least able to spare the
    /// attention. The footer says how far below the end the view is, so a user
    /// who is not following is told what they are missing rather than moved to
    /// it.
    pub scroll: u16,
    pub auto_scroll: bool,
    /// The largest `scroll` the transcript currently allows.
    ///
    /// The wrapped height is only known once the text has been laid out, so
    /// the renderer reports it back here through [`App::set_scroll_max`]. A
    /// `Cell` because the renderer holds `&App`: the value is a measurement of
    /// the last frame, not state the user set.
    scroll_max: std::cell::Cell<u16>,
    /// Frame counter driving the header spinner.
    pub tick: u64,
    /// The run's end of the event channel.
    ///
    /// Bounded, so a fast model cannot outrun a slow terminal. The producer
    /// coalesces rather than blocking on every fragment; see
    /// [`crate::session::drive`].
    pub tx: mpsc::Sender<Event>,
    agent: Agent,
    /// The agent that summarises a conversation, with no tools.
    ///
    /// Held beside the working agent rather than derived from it, because the
    /// guarantee is that the compaction call cannot act: an agent built with
    /// the tools and asked not to use them is not the same thing.
    summariser: Agent,
    /// The session's services: the workspace, the approval gate, the process
    /// runner and the budgets.
    ///
    /// Held so the app can rebuild the agent on a model switch against the
    /// same session, and so the approval channel has somewhere to live. It is
    /// the same value the agent's tools hold, so a switch cannot leave the
    /// new agent pointing at a different workspace than the old one.
    rt: Arc<crate::runtime::Runtime>,
    history: Arc<Mutex<Vec<RigMessage>>>,
    streaming: bool,
    /// Text streamed for each model call of the current run, so a retried
    /// turn's provisional text can be removed exactly.
    turn_text: Vec<String>,
    /// Zero-based index of the model call currently streaming.
    current_turn: usize,
    /// Index of the assistant message the current run writes into.
    assistant_idx: Option<usize>,
    /// Tool blocks the user has expanded, by message index. Collapsed is the
    /// default, so this holds the exceptions rather than the whole set.
    expanded: std::collections::HashSet<usize>,
    /// Tool groups the user has folded, by the group's first message index.
    /// Open is the default: the grouping exists to keep every call visible,
    /// so folding is the exception.
    folded: std::collections::HashSet<usize>,
    /// The `/model` picker, when it is open.
    pub picker: Option<ModelPicker>,
    /// The approval prompt on screen, if the run is waiting on an answer.
    ///
    /// Held rather than answered on arrival: the request is a question for
    /// the user, and the tool that asked is blocked until it is answered.
    /// Dropping it without answering denies the action, which is what the
    /// approver does with a request nobody answered, so quitting with a
    /// prompt up cannot leave a write half-decided.
    pub approval: Option<crate::approval::PendingApproval>,
    /// Where approval requests arrive from the tools.
    ///
    /// Taken by the event loop, which is the only thing that can select on
    /// it. It is `None` in a test that drives the app without a terminal, and
    /// the loop treats that as "no prompt will ever arrive" rather than as an
    /// error.
    approval_rx: Option<mpsc::UnboundedReceiver<crate::approval::PendingApproval>>,
    /// Whether the run in flight has produced an answer.
    ///
    /// The stream handler sets the run's state and the transcript's content in
    /// separate steps, so the state can reach `ready` while the last event is
    /// still on its way to the screen. This is what closes that gap: it is
    /// set with the state and answered by the content, so a header keyed off
    /// both cannot claim a run is in flight once its answer has arrived.
    answered: bool,
    /// The background tasks this app started: the stream task, the model
    /// fetch, the compaction.
    ///
    /// Held so the event loop can wait for them before the terminal is given
    /// back. A task still running when the process exits is a task whose
    /// child process is still running, and a child that outlives the session
    /// keeps writing to a screen nobody owns. The handles are also what makes
    /// the wait possible at all: `tokio::spawn` returns one and dropping it
    /// detaches the task rather than stopping it.
    tasks: Vec<tokio::task::JoinHandle<()>>,
    /// What the run in flight has done, as it does it.
    ///
    /// The loop the model is asked to follow is analyze, gather, plan,
    /// execute, verify, finish. Until this existed that loop was only in the
    /// prompt: the app saw a stream of deltas and tool results and had no idea
    /// which stage the run was in, so it could not say that a change had gone
    /// unverified, and it could not tell a run that was correcting itself from
    /// one that was stuck.
    session: crate::session::Session,
    /// The path each write call named, by call id.
    ///
    /// A tool result carries the tool's name but not its arguments, so the
    /// file a write touched is only knowable from the call that asked for it.
    /// The two arrive as separate events, and calls in one turn run
    /// concurrently, so the pairing is by id rather than by order.
    call_paths: std::collections::HashMap<String, String>,
}

/// How a tool result ended, read from the result itself.
///
/// The result text is all that reaches the transcript: rig hands the UI a
/// [`rig::message::ToolResult`] with a body and an `is_error` flag, and the
/// flag is true for a refusal and a cancellation as well as for a failure.
/// The two refusals are told apart by the text the library writes for them,
/// which is why those texts are constants with tests pinning them.
///
/// A refusal is not a failure. The tool worked, the user said no, and the
/// transcript must not paint that with the same cross as a broken build.
fn outcome_of(body: &str, is_error: bool) -> ToolOutcome {
    if !is_error {
        return ToolOutcome::Ok;
    }
    let text = body.trim_start();
    if text.starts_with(crate::error::DENIED_PREFIX) {
        ToolOutcome::Denied
    } else if text == crate::error::CANCELLED_TEXT {
        ToolOutcome::Cancelled
    } else {
        ToolOutcome::Failed
    }
}

/// The status line's word for a call that has just been committed.
///
/// The tool's name is the mechanism; the act is what the user is watching for.
/// `read_file` and `grep` are both searching, `write_file` and `edit_file` are
/// both changing the project, and a command that checks the change is the run
/// verifying itself rather than doing something new. The categories are the
/// ones the transcript already groups by, so the status line and the transcript
/// agree about what kind of thing is happening.
fn action_status(call: &rig::message::ToolCall) -> String {
    let name = call.function.name.to_string();
    match crate::tools::ToolCategory::of(&name) {
        crate::tools::ToolCategory::Explore => "searching".to_string(),
        crate::tools::ToolCategory::Edit => "applying changes".to_string(),
        crate::tools::ToolCategory::Run => {
            let command = call
                .function
                .arguments
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if crate::session::is_verification_command(command) {
                "running checks".to_string()
            } else {
                format!("running {name}")
            }
        }
        crate::tools::ToolCategory::Other => format!("running {name}"),
    }
}

impl App {
    pub fn new(
        agents: crate::agent::Agents,
        rt: Arc<crate::runtime::Runtime>,
    ) -> (Self, mpsc::Receiver<Event>) {
        let (tx, rx) = mpsc::channel(crate::event::EVENT_QUEUE);
        let model_name = rt.config().model.clone();
        let project_name = rt.workspace().name();
        let crate::agent::Agents { main, summariser } = agents;

        let app = Self {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            status: "ready".to_string(),
            model_name,
            project_name,
            should_quit: false,
            history_input: Vec::new(),
            history_pos: None,
            history_draft: String::new(),
            cancel: None,
            cancelling: false,
            pending: std::collections::VecDeque::new(),
            activity: Vec::new(),
            body_cache: std::cell::RefCell::new(crate::render::BodyCache::default()),
            reasoning: String::new(),
            usage_in: 0,
            usage_out: 0,
            calls: 0,
            context_tokens: 0,
            estimated_tokens: 0,
            recovery_used: false,
            retry_after_compaction: None,
            compacting: false,
            scroll: 0,
            auto_scroll: true,
            scroll_max: std::cell::Cell::new(0),
            tick: 0,
            tx,
            agent: main,
            summariser,
            rt,
            history: Arc::new(Mutex::new(Vec::new())),
            streaming: false,
            turn_text: Vec::new(),
            current_turn: 0,
            assistant_idx: None,
            expanded: std::collections::HashSet::new(),
            folded: std::collections::HashSet::new(),
            picker: None,
            approval: None,
            approval_rx: None,
            answered: false,
            tasks: Vec::new(),
            session: crate::session::Session::new(0),
            call_paths: std::collections::HashMap::new(),
        };
        (app, rx)
    }

    /// Attach the channel approval requests arrive on.
    ///
    /// Called by the binary once the app exists, because the approver needs
    /// the sender and the app needs the receiver: the two are made together
    /// and one of them has to be handed over afterwards.
    pub fn set_approval_channel(
        &mut self,
        rx: mpsc::UnboundedReceiver<crate::approval::PendingApproval>,
    ) {
        self.approval_rx = Some(rx);
    }

    /// Take the approval channel, for the event loop to select on.
    pub fn take_approval_channel(
        &mut self,
    ) -> Option<mpsc::UnboundedReceiver<crate::approval::PendingApproval>> {
        self.approval_rx.take()
    }

    /// Whether the run is waiting for the user to answer a prompt.
    pub fn is_awaiting_approval(&self) -> bool {
        self.approval.is_some()
    }

    /// Answer the prompt on screen.
    ///
    /// A no-op when there is none, so a key that arrives after the answer
    /// cannot answer a second time: the request is taken, and the oneshot
    /// behind it is consumed by the first answer.
    pub fn answer_approval(&mut self, approve: bool) {
        if let Some(pending) = self.approval.take() {
            let decision = if approve {
                crate::approval::Decision::Approved
            } else {
                crate::approval::Decision::Denied("the user declined".to_string())
            };
            pending.answer(decision);
        }
    }

    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Whether the prompt in flight has already spent its one recovery
    /// attempt.
    ///
    /// Read by the tests that pin the retry to exactly one per prompt, and by
    /// anything that wants to say so on screen. The flag is reset when the
    /// user sends a new prompt, so it describes the prompt in flight rather
    /// than the session.
    pub fn recovery_used(&self) -> bool {
        self.recovery_used
    }

    /// Name the project the session is working in, from its workspace root.
    pub fn set_project_name(&mut self, name: impl Into<String>) {
        self.project_name = name.into();
    }

    /// Record that a run is in flight or has finished, with the state word the
    /// header shows: as the run's status, and again beside the answer while it
    /// is being written.
    ///
    /// `streaming` and `status` are one fact seen in two places, so they are
    /// set together: a run that is streaming with the status left at `ready`
    /// would leave the header reading idle while the transcript says
    /// otherwise.
    pub fn set_streaming(&mut self, streaming: bool, status: &str) {
        self.streaming = streaming;
        self.status = status.to_string();
        // The run's state is the stream handler's to set, and the handler
        // cannot have answered yet: it is what produces the answer. So this
        // records that the state and the screen now agree, which is what the
        // transcript header reads when it decides whether the answer has
        // arrived.
        self.answered = !streaming;
    }

    /// Whether the run in flight has produced an answer. See [`App::answered`].
    pub fn has_answered(&self) -> bool {
        self.answered
    }

    /// True between the user asking to stop a run and the run confirming it
    /// has stopped.
    pub fn is_cancelling(&self) -> bool {
        self.cancelling
    }

    // ------------------------------------------------------------ the caret

    /// The caret position, clamped to the buffer.
    ///
    /// Every edit goes through this rather than trusting the stored index, so
    /// the caret can never point past the end of the buffer.
    pub fn caret(&self) -> usize {
        self.cursor.min(self.input.chars().count())
    }

    /// The byte offset of the caret, for slicing.
    fn caret_byte(&self) -> usize {
        self.input
            .char_indices()
            .nth(self.caret())
            .map(|(i, _)| i)
            .unwrap_or(self.input.len())
    }

    /// Replace the buffer and put the caret at the end of it.
    pub fn set_input(&mut self, text: impl Into<String>) {
        self.input = text.into();
        self.cursor = self.input.chars().count();
    }

    /// The prompt being composed.
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Insert one character at the caret.
    pub fn insert_char(&mut self, c: char) {
        let at = self.caret_byte();
        self.input.insert(at, c);
        self.cursor = self.caret() + 1;
    }

    /// Insert a string at the caret, leaving the caret after it.
    ///
    /// A paste arrives as one event and may hold newlines; inserting it
    /// verbatim is what keeps a pasted block from submitting the prompt.
    pub fn insert_str(&mut self, text: &str) {
        let at = self.caret_byte();
        self.input.insert_str(at, text);
        self.cursor = self.caret() + text.chars().count();
    }

    /// Delete the character before the caret.
    pub fn backspace(&mut self) {
        let caret = self.caret();
        if caret == 0 {
            return;
        }
        let start = self
            .input
            .char_indices()
            .nth(caret - 1)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let end = self.caret_byte();
        self.input.replace_range(start..end, "");
        self.cursor = caret - 1;
    }

    /// Delete the character under the caret.
    pub fn delete_forward(&mut self) {
        let caret = self.caret();
        if caret >= self.input.chars().count() {
            return;
        }
        let start = self.caret_byte();
        let end = self
            .input
            .char_indices()
            .nth(caret + 1)
            .map(|(i, _)| i)
            .unwrap_or(self.input.len());
        self.input.replace_range(start..end, "");
    }

    /// Move the caret `delta` characters, clamped to the buffer.
    pub fn move_caret(&mut self, delta: i32) {
        let next = self.caret() as i32 + delta;
        self.cursor = next.clamp(0, self.input.chars().count() as i32) as usize;
    }

    /// Move the caret one word left (`dir < 0`) or right.
    ///
    /// A word is a run of non-whitespace, so the caret lands on the boundary
    /// the user sees rather than on a punctuation class they do not.
    pub fn move_caret_word(&mut self, dir: i32) {
        let chars: Vec<char> = self.input.chars().collect();
        let mut i = self.caret();
        if dir < 0 {
            while i > 0 && chars[i - 1].is_whitespace() {
                i -= 1;
            }
            while i > 0 && !chars[i - 1].is_whitespace() {
                i -= 1;
            }
        } else {
            while i < chars.len() && !chars[i].is_whitespace() {
                i += 1;
            }
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
        }
        self.cursor = i;
    }

    /// The caret's offset within its own logical line.
    pub fn caret_in_line(&self) -> usize {
        let caret = self.caret();
        self.input
            .chars()
            .take(caret)
            .collect::<Vec<char>>()
            .into_iter()
            .rev()
            .take_while(|c| *c != '\n')
            .count()
    }

    /// Move the caret to the start of its logical line.
    pub fn caret_line_start(&mut self) {
        self.move_caret(-(self.caret_in_line() as i32));
    }

    /// Move the caret to the end of its logical line.
    pub fn caret_line_end(&mut self) {
        let rest = self.input.chars().skip(self.caret());
        let to_end = rest.take_while(|c| *c != '\n').count();
        self.move_caret(to_end as i32);
    }

    /// Delete from the caret back to the start of its logical line.
    pub fn kill_to_line_start(&mut self) {
        let n = self.caret_in_line();
        for _ in 0..n {
            self.backspace();
        }
    }

    /// Delete from the caret to the end of its logical line.
    pub fn kill_to_line_end(&mut self) {
        let rest = self.input.chars().skip(self.caret());
        let n = rest.take_while(|c| *c != '\n').count();
        for _ in 0..n {
            self.delete_forward();
        }
    }

    /// Delete the word before the caret, and the whitespace before that.
    pub fn delete_word_before(&mut self) {
        let before = self.caret();
        self.move_caret_word(-1);
        let after = self.caret();
        self.cursor = before;
        for _ in after..before {
            self.backspace();
        }
    }

    /// How many logical lines the input holds.
    pub fn input_lines(&self) -> usize {
        self.input.chars().filter(|c| *c == '\n').count() + 1
    }

    /// The height the input box wants: one row per logical line, plus the
    /// border, capped so a pasted file cannot take the whole screen.
    ///
    /// The caller clamps this against the space it actually has; the box is
    /// the last thing to be given room, not the first.
    pub fn input_height(&self) -> u16 {
        (self.input_lines().min(MAX_INPUT_ROWS as usize) as u16) + 2
    }

    /// The row of the caret, counted in logical lines.
    pub fn caret_line(&self) -> usize {
        self.input
            .chars()
            .take(self.caret())
            .filter(|c| *c == '\n')
            .count()
    }

    // ---------------------------------------------------------- the history

    /// Prompts the user has sent, oldest first, for Up/Down recall.
    pub fn history(&self) -> &[String] {
        &self.history_input
    }

    /// Recall the previous prompt, saving the draft on the way in.
    pub fn history_prev(&mut self) {
        if self.history_input.is_empty() {
            return;
        }
        let next = match self.history_pos {
            None => {
                self.history_draft = self.input.clone();
                self.history_input.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_pos = Some(next);
        self.set_input(self.history_input[next].clone());
    }

    /// Recall the next prompt, restoring the draft past the newest entry.
    pub fn history_next(&mut self) {
        let Some(i) = self.history_pos else {
            return;
        };
        if i + 1 < self.history_input.len() {
            self.history_pos = Some(i + 1);
            self.set_input(self.history_input[i + 1].clone());
        } else {
            self.history_pos = None;
            let draft = std::mem::take(&mut self.history_draft);
            self.set_input(draft);
        }
    }

    /// Record a sent prompt, so Up recalls it.
    ///
    /// A repeat of the newest entry is not stored twice: re-sending the same
    /// prompt is common, and a history full of one line is useless.
    fn remember_prompt(&mut self, text: &str) {
        if self.history_input.last().map(String::as_str) != Some(text) {
            self.history_input.push(text.to_string());
        }
        self.history_pos = None;
        self.history_draft.clear();
    }

    // -------------------------------------------------------- cancellation

    /// Ask the run in flight to stop.
    ///
    /// The stream task is woken by the channel and drops the rig stream, which
    /// drops the tool future in flight; a `shell` child is killed by its
    /// `kill_on_drop`. The UI stays in `cancelling` until the task confirms,
    /// so a second Esc is not needed to get the keyboard back.
    pub fn cancel_run(&mut self) {
        if !self.streaming || self.cancelling {
            return;
        }
        if let Some(signal) = self.cancel.as_ref() {
            signal.stop();
        }
        // The session's own signal as well: it is what the tools and the
        // search worker watch, and a stop that only reached the stream would
        // leave a `grep` walking a large tree until it finished.
        self.rt.cancel().stop();
        self.cancelling = true;
        self.status = "cancelling".to_string();
    }

    /// How many prompts are waiting for the run in flight to settle.
    pub fn queued(&self) -> usize {
        self.pending.len()
    }

    /// Start the next queued prompt, if there is one.
    fn drain_pending(&mut self) {
        // A run cannot start while a compaction is in flight, and a prompt
        // popped here would be handed to a `send_message` that re-queues it,
        // which moves it to the back: two prompts typed ahead would run in
        // the wrong order. Leave the queue alone and let `Event::Compacted`
        // drain it once the summary has landed.
        if self.compacting {
            return;
        }
        if let Some(next) = self.pending.pop_front() {
            self.send_message(next);
        }
    }

    /// The prompt the run in flight was started with: the newest user
    /// message, since a queued prompt is not in the transcript yet.
    fn last_user_prompt(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
    }

    /// Send a user message: append it, then stream the agent's reply in the
    /// background, translating every stream item into an [`Event`].
    pub fn send_message(&mut self, text: String) {
        // A new prompt gets its own recovery attempt: the one the previous
        // prompt spent was against the history as it stood then.
        self.recovery_used = false;
        self.start_run(text);
    }

    /// Start a run for `text`, appending it to the transcript.
    ///
    /// Split from [`App::send_message`] so the recovery retry can start a run
    /// without clearing the flag that says the attempt has been spent.
    fn start_run(&mut self, text: String) {
        let trimmed = text.trim().to_string();
        if trimmed.is_empty() {
            return;
        }

        if trimmed.starts_with('/') {
            self.handle_command(&trimmed);
            return;
        }

        // A run started mid-compaction would replay the pre-compaction
        // history and silently undo the summary, so the prompt waits. It is
        // queued rather than refused because the input box has already been
        // cleared by the time this is called: refusing would lose what the
        // user typed, and the queue is drained the moment the summary lands.
        if self.compacting {
            self.pending.push_back(trimmed);
            self.messages.push(ChatMessage::new(
                Role::System,
                format!(
                    "Compacting the conversation; queued ({} waiting).",
                    self.pending.len()
                ),
            ));
            self.auto_scroll = true;
            return;
        }

        // A run is already in flight: queue this one rather than starting a
        // second. Two runs at once would interleave into one transcript and
        // both write into the same assistant message.
        if self.streaming {
            self.pending.push_back(trimmed);
            self.messages.push(ChatMessage::new(
                Role::System,
                format!(
                    "Queued ({} waiting). Esc stops the run in flight.",
                    self.pending.len()
                ),
            ));
            self.auto_scroll = true;
            return;
        }

        self.messages
            .push(ChatMessage::new(Role::User, trimmed.clone()));
        self.messages
            .push(ChatMessage::new(Role::Assistant, String::new()));
        self.assistant_idx = Some(self.messages.len() - 1);
        self.status = "thinking".to_string();
        self.streaming = true;
        self.cancelling = false;
        self.auto_scroll = true;
        self.activity.clear();
        self.reasoning.clear();
        self.turn_text.clear();
        self.current_turn = 0;
        self.remember_prompt(&trimmed);

        // A fresh run id, so every action the tools record from here on is
        // attributed to this run rather than to the session. The session
        // recorder starts empty with it: what the *previous* run did is in
        // the journal, and carrying it forward would make this run's outcome
        // claim changes it never made.
        let run = self.rt.begin_run();
        self.session = crate::session::Session::new(run);
        self.emit_stage(crate::session::Stage::Analyze);

        let agent = self.agent.clone();
        let history = self.history.clone();
        let tx = self.tx.clone();
        // The session's services, so the run reads its turn budget, its idle
        // timeout and its repair counters from the same resolved configuration
        // the rest of the app does. Reading them from the environment inside
        // `drive` is what made `LLM_MAX_TURNS` and the configured limit two
        // different numbers.
        let rt = self.rt.clone();

        // A fresh signal per run, so a stop from an earlier run can never be
        // read as a stop for this one.
        let cancel = CancelToken::new();
        self.cancel = Some(cancel.clone());
        // The session's signal is cleared for the same reason, and because it
        // is the one the tools check before they commit anything: a stop left
        // over from the last run would refuse the first write of this one.
        self.rt.cancel().reset();

        let handle = tokio::spawn(async move {
            // A panic anywhere in the run would otherwise be invisible: the
            // task dies, the channel goes quiet, and the app waits forever on
            // a stream that will never send anything. Catching it here turns
            // that into a terminal event, which is the difference between a
            // run that failed and a run that appears to still be thinking.
            let failure_tx = tx.clone();
            let run_body = crate::session::drive(agent, history, trimmed.clone(), cancel, tx, rt);
            if let Err(panic) = std::panic::AssertUnwindSafe(run_body).catch_unwind().await {
                let _ = failure_tx
                    .send(Event::TaskFailed(panic_message(&panic)))
                    .await;
            }
        });
        self.tasks.push(handle);
    }

    /// Open the `/model` picker and start fetching the list.
    ///
    /// `ollama list` is a subprocess, so it runs on its own task and the
    /// result arrives as an event; the picker opens immediately with a
    /// "loading" state rather than freezing the UI on the call. It goes
    /// through the session's process runner like every other command, so it
    /// has the same deadline, the same output bound and the same kill path,
    /// and it is watched by the run's cancel token.
    fn open_model_picker(&mut self) {
        self.picker = Some(ModelPicker {
            loading: true,
            ..Default::default()
        });
        let tx = self.tx.clone();
        let rt = self.rt.clone();
        let handle = tokio::spawn(async move {
            let result = crate::agent::list_ollama_models(&rt)
                .await
                .map_err(|e| e.to_string());
            let _ = tx.send(Event::ModelsLoaded(result)).await;
        });
        self.tasks.push(handle);
    }

    /// Close the picker without changing the model.
    pub fn close_picker(&mut self) {
        self.picker = None;
    }

    /// Switch to the highlighted model, rebuilding the agent.
    ///
    /// The conversation is kept: the history is carried by the agent, so it is
    /// replayed into the new one rather than dropped.
    pub fn confirm_model(&mut self) {
        let Some(model) = self
            .picker
            .as_ref()
            .and_then(|p| p.current())
            .map(str::to_string)
        else {
            return;
        };
        self.picker = None;

        match crate::agent::create_agents_with_model(&model, &self.rt) {
            Ok(agents) => {
                self.agent = agents.main;
                self.summariser = agents.summariser;
                self.model_name = model.clone();
                self.messages.push(ChatMessage::new(
                    Role::System,
                    format!("Switched to {model}."),
                ));
                self.auto_scroll = true;
            }
            Err(e) => {
                self.messages.push(ChatMessage::new(
                    Role::System,
                    format!("Could not switch to {model}: {e}"),
                ));
                self.auto_scroll = true;
            }
        }
    }

    /// Summarise the conversation and replace the model's history with the
    /// summary, followed by the most recent turns kept verbatim.
    ///
    /// Only what the model carries forward is rewritten: the visible
    /// transcript is left alone, so the user keeps their scrollback and a
    /// `system` row marks where the model's memory was compacted. The
    /// summarisation is one model call, sent with the evicted turns attached,
    /// so the model already has everything it needs to summarise.
    pub fn compact(&mut self) {
        if self.compacting || self.streaming {
            return;
        }

        let history = self.history.lock().map(|h| h.clone()).unwrap_or_default();
        if history.is_empty() {
            self.messages.push(ChatMessage::new(
                Role::System,
                "Nothing to compact: the conversation is empty.".to_string(),
            ));
            self.auto_scroll = true;
            return;
        }

        // The tail is kept verbatim, so the summariser is sent only what is
        // being evicted. That is what makes the summary a memory of the
        // evicted turns rather than a second copy of the ones that survive,
        // and it is what anchors the summary: the previous summary is part of
        // the evicted prefix, so each one is built on the last instead of
        // being re-derived from a conversation that no longer exists.
        let budget = self.config().context.tail_chars;
        let evicted = history[..crate::agent::tail_start(&history, budget)].to_vec();

        self.compacting = true;
        self.streaming = true;
        self.status = "compacting".to_string();
        self.auto_scroll = true;

        // The summariser, not the working agent: it has no tools, so the one
        // call that is handed a whole conversation cannot change anything
        // while it reads it.
        let agent = self.summariser.clone();
        let tx = self.tx.clone();

        // The session's stop signal, which is what the tools and the search
        // worker watch. A compaction is a model call like any other and has to
        // be stoppable: it is the longest single call the app makes, and a
        // user who presses Esc while it runs means it. `cancel_run` stops this
        // same token, so the two paths meet here rather than in a second
        // signal that nothing would watch.
        let stop = self.rt.cancel().clone();
        stop.reset();

        let handle = tokio::spawn(async move {
            // `max_turns(1)` bounds this to a single model call: the prompt
            // asks for the summary and the agent has no tools, so a second
            // call would only mean the model went exploring instead of
            // answering.
            //
            // Driven as a stream rather than awaited, so it takes the same
            // path as a normal run: one shape of rig call in the app, and the
            // tests can script both from the same fixture.
            let mut stream = agent
                .prompt(crate::agent::COMPACT_REQUEST)
                .history(evicted)
                .max_turns(1)
                .stream();

            let mut result = Err("the compaction run produced no response".to_string());
            loop {
                let item = tokio::select! {
                    biased;
                    _ = stop.notified() => {
                        if stop.is_stopped() {
                            result = Err("cancelled".to_string());
                            break;
                        }
                        continue;
                    }
                    item = stream.next() => match item {
                        Some(item) => item,
                        None => break,
                    },
                };
                match item {
                    Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                        result = Ok(response.output());
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        result = Err(e.to_string());
                        break;
                    }
                }
            }

            let _ = tx.send(Event::Compacted(result)).await;
        });
        self.tasks.push(handle);
    }

    /// Stop the run in flight and wait, briefly, for the background tasks to
    /// finish.
    ///
    /// Called on the way out of the event loop, before the terminal is given
    /// back. The order matters: a task still running holds a child process,
    /// and a child that outlives the session keeps writing to a screen that
    /// is no longer ours. Stopping first is what makes the wait short, since
    /// the tasks are watching the same signal.
    ///
    /// The wait is bounded. A task that will not stop is a task that would
    /// otherwise hang the exit, and the process is going away regardless; the
    /// count of what did not finish is returned so the caller can say so
    /// rather than pretend the shutdown was clean.
    pub async fn settle(&mut self, grace: std::time::Duration) -> usize {
        self.cancel_run();
        // A run that is not streaming has no signal to stop, but a compaction
        // or a model fetch may still be in flight, so the tasks are always
        // waited for.
        let tasks = std::mem::take(&mut self.tasks);
        let mut unfinished = 0;
        for task in tasks {
            // The handle is taken before the wait because the wait consumes
            // it: a task that outlives the grace period has to be aborted, and
            // dropping a `JoinHandle` detaches rather than stops.
            let abort = task.abort_handle();
            if tokio::time::timeout(grace, task).await.is_err() {
                abort.abort();
                unfinished += 1;
            }
        }
        unfinished
    }

    /// How many messages the model's history holds.
    pub fn history_len(&self) -> usize {
        self.history.lock().map(|h| h.len()).unwrap_or(0)
    }

    /// The slash commands the TUI understands, in `/help` order.
    pub const SLASH_COMMANDS: &'static [&'static str] =
        &["/clear", "/compact", "/exit", "/help", "/model", "/quit"];

    /// Whether `input` is currently a bare slash-command token with no
    /// arguments after it, i.e. something that can be autocompleted.
    fn is_slash_command_token(input: &str) -> bool {
        input.starts_with('/')
            && !input.trim().contains(char::is_whitespace)
            && !input.contains('\n')
    }

    /// Autocomplete the command name in `self.input`. Returns true if the
    /// input changed.
    pub fn autocomplete(&mut self) -> bool {
        self.complete_slash_command()
    }

    pub fn complete(&mut self) -> bool {
        self.complete_slash_command()
    }

    pub fn complete_command(&mut self) -> bool {
        self.complete_slash_command()
    }

    pub fn autocomplete_command(&mut self) -> bool {
        self.complete_slash_command()
    }

    pub fn complete_slash(&mut self) -> bool {
        self.complete_slash_command()
    }

    pub fn completions(&self) -> Vec<&'static str> {
        self.command_suggestions()
    }

    pub fn suggestions(&self) -> Vec<&'static str> {
        self.command_suggestions()
    }

    pub fn command_completions(&self) -> Vec<&'static str> {
        self.command_suggestions()
    }

    pub fn autocomplete_suggestions(&self) -> Vec<&'static str> {
        self.command_suggestions()
    }

    pub fn command_suggestions(&self) -> Vec<&'static str> {
        if !Self::is_slash_command_token(&self.input) {
            return Vec::new();
        }
        let prefix = self.input.to_lowercase();
        Self::SLASH_COMMANDS
            .iter()
            .copied()
            .filter(|c| c.to_lowercase().starts_with(&prefix))
            .collect()
    }

    pub fn complete_slash_command(&mut self) -> bool {
        if !Self::is_slash_command_token(&self.input) {
            return false;
        }
        let prefix = self.input.to_lowercase();
        let matches: Vec<&str> = Self::SLASH_COMMANDS
            .iter()
            .copied()
            .filter(|c| c.to_lowercase().starts_with(&prefix))
            .collect();
        if matches.is_empty() {
            return false;
        }
        if matches.len() == 1 {
            if self.input == matches[0] {
                return false;
            }
            self.set_input(matches[0]);
            return true;
        }
        // The shared prefix is built as a string, not as a length, so the
        // index into `first` is always a character boundary. `common` counts
        // characters while `first.len()` counts bytes, and the two only agree
        // while every command happens to be ASCII; slicing on the character
        // count would panic the moment one is not.
        let first = matches[0];
        let mut common = first.chars().count();
        for m in matches.iter().skip(1) {
            let c = first
                .chars()
                .zip(m.chars())
                .take_while(|(a, b)| a.eq_ignore_ascii_case(b))
                .count();
            common = common.min(c);
        }
        let shared: String = first.chars().take(common).collect();
        if shared.chars().count() > self.input.chars().count() {
            self.set_input(shared);
            return true;
        }
        false
    }
    fn handle_command(&mut self, cmd: &str) {
        match cmd {
            "/model" => self.open_model_picker(),
            "/compact" => self.compact(),
            "/help" => {
                self.messages
                    .push(ChatMessage::new(Role::System, HELP_TEXT.to_string()));
                self.auto_scroll = true;
            }
            "/clear" => {
                self.messages.clear();
                self.activity.clear();
                self.reasoning.clear();
                self.turn_text.clear();
                self.current_turn = 0;
                self.assistant_idx = None;
                self.expanded.clear();
                self.folded.clear();
                self.usage_in = 0;
                self.usage_out = 0;
                self.calls = 0;
                self.context_tokens = 0;
                self.scroll = 0;
                self.pending.clear();
                if let Ok(mut hist) = self.history.lock() {
                    hist.clear();
                }
                self.status = "ready".to_string();
            }
            "/quit" | "/exit" => {
                self.should_quit = true;
            }
            other => {
                self.messages.push(ChatMessage::new(
                    Role::System,
                    format!("Unknown command: {other}. Type /help for a list."),
                ));
                self.auto_scroll = true;
            }
        }
    }

    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::TextDelta(text) => {
                while self.turn_text.len() <= self.current_turn {
                    self.turn_text.push(String::new());
                }
                // A turn that follows one which already produced text starts
                // a new paragraph. Without this the two runs of text are
                // concatenated mid-sentence ("...one.Now let me...").
                let first_of_turn = self.turn_text[self.current_turn].is_empty();
                if first_of_turn && self.current_turn > 0 && !self.assistant_is_empty() {
                    self.append_assistant("\n\n");
                    self.turn_text[self.current_turn].push_str("\n\n");
                }
                self.append_assistant(&text);
                self.turn_text[self.current_turn].push_str(&text);
                self.status = "streaming".to_string();
            }

            Event::ReasoningDelta(text) => {
                self.reasoning.push_str(&text);
                self.status = "reasoning".to_string();
            }

            Event::ToolCallStreaming { name, part } => {
                self.push_activity(Activity {
                    kind: ActivityKind::ToolCall,
                    title: format!("{name} (streaming arguments)"),
                    detail: String::new(),
                    is_error: false,
                    part: Some(part),
                });
                // The block appears in the transcript as soon as the call
                // opens, so the conversation shows what is happening while the
                // arguments are still arriving.
                self.push_tool_block(ToolBlock {
                    name: name.clone(),
                    streaming: true,
                    part: Some(part),
                    ..Default::default()
                });
                self.status = format!("calling {name}");
            }

            Event::ToolCallIdentified { part, call } => {
                // The part ended and named its call. Binding the placeholder
                // here rather than at the commit is what makes the association
                // exact: a turn can open a part for a call that is then
                // rejected, and the commit that follows belongs to a later
                // part, not to the next block in the transcript.
                //
                // Only an unbound placeholder is a candidate. Part positions
                // restart at zero every turn, so a block that already carries a
                // call is a previous turn's, and binding this turn's call to it
                // would rename a call that has already run.
                if let Some(block) =
                    self.waiting_tool_block_mut(|b| b.call.is_none() && b.part == Some(part))
                {
                    block.call = Some(call);
                }
            }

            Event::ToolArgsDelta { part, json } => {
                // The fragment belongs to one part, and a turn can stream
                // several at once. The block for this part keeps the whole
                // raw document so the summary can be rebuilt as it grows;
                // the newest block is not necessarily this one.
                let detail = self
                    .waiting_tool_block_mut(|b| b.streaming && b.part == Some(part))
                    .map(|block| {
                        block.args_json.push_str(&json);
                        let args = rig::streaming::parse_partial_arguments(&block.args_json);
                        let detail = summarize_args(&args);
                        block.args = detail.clone();
                        detail
                    });
                if let Some(detail) = detail {
                    if let Some(row) = self
                        .activity
                        .iter_mut()
                        .rev()
                        .find(|a| a.kind == ActivityKind::ToolCall && a.part == Some(part))
                    {
                        row.detail = detail;
                    }
                }
            }

            Event::ToolCallStarted(call) => {
                let name = call.function.name.to_string();
                let detail = summarize_args(&call.function.arguments);
                let call_id = call.id.to_string();
                // The run's own record of what it is doing, before the
                // transcript is touched: the stage is a function of the calls
                // that have been made, and this is the call.
                self.note_call(&call);
                // The block bound to this call knows which part opened it.
                // That is the activity row's identity while two calls of the
                // same name are streaming at once; the title alone cannot
                // tell them apart.
                let part = self
                    .messages
                    .iter()
                    .filter_map(|m| m.tool.as_ref())
                    .find(|b| b.call.as_deref() == Some(call_id.as_str()))
                    .and_then(|b| b.part);
                // The streaming placeholder for this call is superseded by
                // the committed call. It is not necessarily the last row: a
                // `Completion` for the turn that emitted the call lands
                // between the two, so search back for it.
                let placeholder = format!("{name} (streaming arguments)");
                let superseded = part
                    .and_then(|part| {
                        self.activity.iter().rposition(|a| {
                            a.kind == ActivityKind::ToolCall
                                && a.part == Some(part)
                                && a.title == placeholder
                        })
                    })
                    .or_else(|| {
                        self.activity.iter().rposition(|a| {
                            a.kind == ActivityKind::ToolCall && a.title == placeholder
                        })
                    });
                match superseded {
                    Some(idx) => {
                        self.activity[idx].title = name.clone();
                        self.activity[idx].detail = detail.clone();
                    }
                    None => self.push_activity(Activity {
                        kind: ActivityKind::ToolCall,
                        title: name.clone(),
                        detail: detail.clone(),
                        is_error: false,
                        part: None,
                    }),
                }
                // The committed call replaces the streaming placeholder's
                // arguments with the validated ones. The placeholder was bound
                // to this call when its part ended, so it is found by id. A
                // provider that never ended the part leaves the block unbound,
                // and then the oldest block still waiting for a call is the
                // right one: a turn can emit several calls, and the stream
                // opens a placeholder for each before any of them commits, so
                // filling the newest would orphan every earlier placeholder.
                let block = self.call_block_mut(&call_id, |b| b.streaming);
                match block {
                    Some(block) => {
                        block.name = name.clone();
                        block.args = detail;
                        block.args_json.clear();
                        block.streaming = false;
                        block.call = Some(call_id);
                    }
                    None => {
                        self.push_tool_block(ToolBlock {
                            name: name.clone(),
                            args: detail,
                            call: Some(call_id),
                            ..Default::default()
                        });
                    }
                }
                self.status = action_status(&call);
            }

            Event::ToolExecuted(call) => {
                let name = call.function.name.to_string();
                self.push_activity(Activity {
                    kind: ActivityKind::ToolExecuted,
                    title: format!("{name} executed"),
                    detail: summarize_args(&call.function.arguments),
                    is_error: false,
                    part: None,
                });
            }

            Event::ToolResult(result) => {
                let name = result.name.to_string();
                let body = result
                    .content
                    .iter()
                    .filter_map(|c| c.as_text())
                    .collect::<Vec<_>>()
                    .join("\n");
                // The run's record first: whether this was a check, and
                // whether it passed, is what the outcome turns on.
                self.note_result(&result);
                self.push_activity(Activity {
                    kind: ActivityKind::ToolResult,
                    title: format!("{name} result"),
                    detail: truncate(&body, 200),
                    is_error: result.is_error,
                    part: None,
                });
                // The result lands in the block for the call it answers. The
                // id is what makes that exact: results are surfaced in call
                // order, but a batch that is interrupted surfaces none, and a
                // block whose call was skipped by a hook never committed, so
                // position alone would put a result on the wrong row. If the
                // call never opened a block, make one now so the result is not
                // lost.
                let call_id = result.call.to_string();
                let outcome = outcome_of(&body, result.is_error);
                let truncated = body.contains(crate::tools::ELISION_MARKER);
                let block = self.call_block_mut(&call_id, |b| !b.done);
                match block {
                    Some(block) => {
                        block.result = body;
                        block.outcome = outcome;
                        block.truncated = truncated;
                        block.done = true;
                        block.streaming = false;
                        block.call = Some(call_id);
                    }
                    None => {
                        self.push_tool_block(ToolBlock {
                            name,
                            result: body,
                            outcome,
                            truncated,
                            done: true,
                            call: Some(call_id),
                            ..Default::default()
                        });
                    }
                }
                // A failed result is the model's cue to correct itself, and
                // saying so is more use than the generic "thinking": the user
                // watching a run that keeps failing wants to know the agent
                // knows, and a run that is correcting is not the same as one
                // that is stuck. A refusal is not a failure and must not be
                // announced as one: the model is told not to retry it, and the
                // user is told their decision was taken.
                self.status = match outcome {
                    ToolOutcome::Failed => "correcting".to_string(),
                    ToolOutcome::Denied => "denied".to_string(),
                    ToolOutcome::Cancelled => "cancelled".to_string(),
                    _ => "thinking".to_string(),
                };
                // A result is the only thing that grows the transcript's
                // memory, so this is where the cap is enforced.
                self.release_old_bodies();
            }

            Event::Completion(call) => {
                self.calls += 1;
                self.usage_in += call.usage.input_tokens.unwrap_or(0);
                self.usage_out += call.usage.output_tokens.unwrap_or(0);
                // The context size is this call's prompt length, not the
                // running total: each call re-sends the whole conversation,
                // so the latest figure is the one the window is measured
                // against.
                if let Some(i) = call.usage.input_tokens {
                    self.context_tokens = i;
                }
                // The estimate the budget hook made for this very request,
                // which is the only place it is measured. Shown next to the
                // provider's report so the two are never confused: this one
                // is an estimate of a request, that one is the exact size of
                // the call that has just been answered.
                if let Some(estimate) = self.rt.last_estimate() {
                    self.estimated_tokens = estimate.tokens as u64;
                }
                let finish = call
                    .finish_reason
                    .as_ref()
                    .map(|f| format!("{f:?}"))
                    .unwrap_or_else(|| "unknown".to_string());
                let tokens = match (call.usage.input_tokens, call.usage.output_tokens) {
                    (Some(i), Some(o)) => format!("{i} in / {o} out"),
                    _ => "usage not reported".to_string(),
                };
                self.push_activity(Activity {
                    kind: ActivityKind::Completion,
                    title: format!("call {} finished ({finish})", call.call_index + 1),
                    detail: tokens,
                    is_error: false,
                    part: None,
                });
                // The next model call of this run streams into a fresh buffer.
                self.current_turn += 1;
                // The provider call is over, so every part of it has ended and
                // every call it was going to name has been named. A
                // placeholder still unbound belongs to a call rig dropped, and
                // it must not be left waiting: part positions restart at zero
                // each turn, so the next turn's first part would bind to it and
                // the transcript would show the wrong call.
                self.close_unbound_placeholders();
            }

            Event::TurnRetried { turn } => {
                self.discard_turn(turn);
                self.push_activity(Activity {
                    kind: ActivityKind::Retry,
                    title: format!("turn {turn} rejected, retrying"),
                    detail: "streamed text for that turn was discarded".to_string(),
                    is_error: true,
                    part: None,
                });
                self.status = "retrying".to_string();
            }

            Event::Finished(response) => {
                self.streaming = false;
                self.cancelling = false;
                self.cancel = None;
                self.status = "ready".to_string();
                self.usage_in = response.usage.input_tokens.unwrap_or(self.usage_in);
                self.usage_out = response.usage.output_tokens.unwrap_or(self.usage_out);
                self.calls = response.completion_calls.len().max(self.calls);
                // The run's final prompt length is the context the next run
                // would start from, so it is the figure the threshold is
                // measured against. `usage` is the run's aggregate and would
                // overstate it.
                if let Some(i) = response
                    .completion_calls
                    .last()
                    .and_then(|c| c.usage.input_tokens)
                {
                    self.context_tokens = i;
                }
                // A run that produced no streamed text (for example one that
                // finalized through an output tool) still has an answer.
                if self.assistant_is_empty() {
                    let output = response.output();
                    if !output.trim().is_empty() {
                        self.append_assistant(&output);
                    }
                }
                self.reasoning.clear();
                self.turn_text.clear();
                self.current_turn = 0;

                // What the run did, said out loud. A run that changed files
                // and never checked them is the failure this whole stage
                // exists to make visible: without it the transcript shows a
                // confident answer and nothing says the change was never run.
                self.report_outcome(crate::session::Terminal::Finished);

                // Compaction is checked here, after the run has settled, so
                // the summary is built from a complete history rather than
                // one a run is still appending to.
                if !self.compacting
                    && crate::agent::should_compact(
                        self.context_tokens,
                        self.config().context.window,
                        self.config().context.compact_at,
                    )
                {
                    self.compact();
                }

                // A prompt typed while this run was in flight goes now, so
                // the queue drains in the order it was filled.
                self.drain_pending();
            }

            Event::ModelsLoaded(result) => {
                if let Some(picker) = self.picker.as_mut() {
                    picker.loading = false;
                    match result {
                        Ok(models) => {
                            // Start on the model in use, so the picker opens
                            // where the user already is.
                            picker.selected = models
                                .iter()
                                .position(|m| *m == self.model_name)
                                .unwrap_or(0);
                            picker.models = models;
                            picker.error = None;
                        }
                        Err(e) => picker.error = Some(e),
                    }
                }
            }

            Event::Compacted(result) => {
                self.compacting = false;
                self.streaming = false;
                // The compaction run is over, so there is nothing left to
                // stop: a stop pressed while it was in flight must not be
                // read as a stop for the next run.
                self.cancelling = false;
                self.cancel = None;
                self.status = "ready".to_string();

                // Whether the history actually shrank. A retry waiting on
                // this compaction is only worth making if it did: retrying
                // against the same history would be rejected the same way.
                let mut compacted = false;

                match result {
                    Ok(summary) if !summary.trim().is_empty() => {
                        compacted = true;
                        let before = self.history_len();
                        let budget = self.config().context.tail_chars;
                        let mut kept = 0;
                        if let Ok(mut hist) = self.history.lock() {
                            // The history is frozen while a compaction is in
                            // flight, because a prompt sent during one is
                            // queued rather than run, so this selects the same
                            // tail `compact` sent the summariser everything
                            // before.
                            let start = crate::agent::tail_start(&hist, budget);
                            let tail = hist[start..].to_vec();
                            let rebuilt = crate::agent::rebuild_history(tail, &summary, budget);
                            // The summary and the acknowledgement are not kept
                            // messages: the first is what replaced the evicted
                            // turns, the second is the line that keeps the
                            // summary and the tail from being two consecutive
                            // user turns. Counting either would tell the user
                            // more survived than did. The acknowledgement is
                            // present exactly when the tail is.
                            kept = rebuilt.len() - 1 - usize::from(rebuilt.len() > 1);
                            *hist = rebuilt;
                        }
                        // The next run reports the real figure; until then the
                        // context size is not known, and a stale one would
                        // re-trigger compaction immediately.
                        self.context_tokens = 0;
                        self.messages.push(ChatMessage::new(
                            Role::System,
                            if kept == 0 {
                                format!("Compacted {before} messages into a summary.")
                            } else {
                                format!(
                                    "Compacted {before} messages into a summary, keeping \
                                     the last {kept}."
                                )
                            },
                        ));
                    }
                    // A failed or empty summarisation must not cost the user
                    // their conversation: being over the threshold is a far
                    // smaller problem than losing the history to it.
                    Ok(_) => {
                        self.messages.push(ChatMessage::new(
                            Role::System,
                            "Compaction produced no summary; the conversation is unchanged."
                                .to_string(),
                        ));
                    }
                    Err(e) => {
                        // A cancellation is not a failure of the summariser,
                        // and saying so would send the user looking for a
                        // problem that is not there.
                        let what = if e == "cancelled" {
                            "Compaction cancelled".to_string()
                        } else {
                            format!("Compaction failed: {e}")
                        };
                        self.messages.push(ChatMessage::new(
                            Role::System,
                            format!("{what}. The conversation is unchanged."),
                        ));
                    }
                }

                // A retry that was waiting on this compaction goes now, and
                // before the queue: it is the prompt whose request was
                // rejected, and it was typed before anything queued behind it.
                if let Some(prompt) = self.retry_after_compaction.take() {
                    if compacted {
                        self.start_run(prompt);
                        return;
                    }
                    self.messages.push(ChatMessage::new(
                        Role::System,
                        "The conversation could not be compacted, so the request was not \
                         retried. Start a new conversation, or raise LLM_NUM_CTX if the \
                         model's window is larger than that."
                            .to_string(),
                    ));
                }

                // Prompts typed while the summary was being written were
                // queued rather than refused, so they go now, in order.
                self.drain_pending();
            }

            Event::Cancelled => {
                self.streaming = false;
                self.cancelling = false;
                self.cancel = None;
                self.status = "ready".to_string();
                self.reasoning.clear();
                self.turn_text.clear();
                self.current_turn = 0;
                // A call that was in flight when the run was abandoned never
                // returned a result, so nothing will ever fill its block in.
                // Left as it stands it would read as still running for the
                // rest of the session, which is the one thing a cancelled
                // block must not say.
                self.mark_in_flight_cancelled();
                // The prompt was delivered, so the model keeps it: the next
                // run continues from what the user said rather than from
                // before it. The answer was cut short, so none of it is
                // carried forward.
                if let Some(text) = self.last_user_prompt() {
                    if let Ok(mut hist) = self.history.lock() {
                        hist.push(RigMessage::user(text));
                    }
                }
                self.messages
                    .push(ChatMessage::new(Role::System, "Run cancelled.".to_string()));
                self.report_outcome(crate::session::Terminal::Cancelled);
                self.drain_pending();
            }

            Event::StreamEnded => {
                self.streaming = false;
                self.cancelling = false;
                self.cancel = None;
                self.status = "error".to_string();
                self.reasoning.clear();
                self.turn_text.clear();
                self.current_turn = 0;
                self.messages.push(ChatMessage::new(
                    Role::System,
                    "The model's stream ended without a final response. The run is over; \
                     anything it changed is on disk."
                        .to_string(),
                ));
                self.report_outcome(crate::session::Terminal::Failed(
                    "the stream ended without a final response".to_string(),
                ));
                self.drain_pending();
            }

            Event::StreamRetry { reason } => {
                // The run is still in flight: the retry happens inside the
                // same task, so `streaming` stays true and the spinner keeps
                // turning. What the user needs is the reason, because a run
                // that quietly re-sends a prompt is a run whose timing they
                // cannot account for.
                self.activity.push(Activity {
                    kind: ActivityKind::Retry,
                    title: "model call failed, retrying".to_string(),
                    detail: reason,
                    is_error: false,
                    part: None,
                });
                self.status = "retrying".to_string();
            }

            Event::TaskFailed(why) => {
                self.streaming = false;
                self.cancelling = false;
                self.cancel = None;
                self.status = "error".to_string();
                self.messages.push(ChatMessage::new(
                    Role::System,
                    format!("The run's task stopped unexpectedly: {why}"),
                ));
                self.report_outcome(crate::session::Terminal::Failed(why));
                self.drain_pending();
            }

            Event::Error(e) => {
                self.streaming = false;
                self.cancelling = false;
                self.cancel = None;
                self.status = "error".to_string();
                // A run that used its whole budget is not something the user
                // can act on from the raw text: rig says "reached the max
                // turns limit of N", which names the number but not the knob.
                // Naming `LLM_MAX_TURNS` is what turns a dead end into a next
                // step, and the transcript is kept, so raising it and sending
                // the prompt again does not lose the work already done.
                let text = if crate::config::is_turn_limit(&e) {
                    format!(
                        "The run used its whole model-call budget of {} \
                         (LLM_MAX_TURNS). Raise it and send the prompt again: \
                         the conversation so far is kept.",
                        self.rt.config().limits.max_turns
                    )
                } else {
                    format!("Error: {e}")
                };
                self.messages.push(ChatMessage::new(Role::System, text));
                self.report_outcome(crate::session::Terminal::Failed(e.clone()));

                // A request the provider rejected for its size is the one
                // failure compaction can fix, so it gets one attempt. The
                // guard is the journal: a run that has already changed files
                // must not be replayed, because the retry would re-issue the
                // calls that made those changes. A run that only read is safe
                // to redo, and redoing it against a smaller history is the
                // whole point.
                if crate::context::is_context_rejection(&e)
                    && !self.recovery_used
                    && self.rt.actions().is_empty()
                    && self.history_len() > 0
                {
                    if let Some(prompt) = self.last_user_prompt() {
                        self.recovery_used = true;
                        self.retry_after_compaction = Some(prompt);
                        self.messages.push(ChatMessage::new(
                            Role::System,
                            "The request was too large for the model's window. \
                             Compacting the conversation and retrying once."
                                .to_string(),
                        ));
                        self.compact();
                        return;
                    }
                }

                self.drain_pending();
            }
        }
    }

    /// The index of the tool block for the call with this id, if one is open.
    fn tool_block_index_by_call(&self, call: &str) -> Option<usize> {
        self.messages.iter().position(|m| {
            m.tool
                .as_ref()
                .is_some_and(|b| b.call.as_deref() == Some(call))
        })
    }

    /// The index of the oldest tool block still waiting for `waiting`, if any.
    fn waiting_tool_block_index(&self, waiting: impl Fn(&ToolBlock) -> bool) -> Option<usize> {
        self.messages
            .iter()
            .position(|m| m.tool.as_ref().is_some_and(&waiting))
    }

    /// The block a call's commit or result belongs to.
    ///
    /// The id is the identity the execution commit and the result share, so
    /// the block bound to `call` is the right one. A provider that never ended
    /// the call's part leaves it unbound, and then the oldest block still
    /// waiting for `waiting` is: a turn can emit several calls, and the stream
    /// opens a placeholder for each before any of them commits, so the newest
    /// block is the wrong one as soon as a turn makes two calls.
    fn call_block_mut(
        &mut self,
        call: &str,
        waiting: impl Fn(&ToolBlock) -> bool,
    ) -> Option<&mut ToolBlock> {
        let idx = self
            .tool_block_index_by_call(call)
            .or_else(|| self.waiting_tool_block_index(waiting))?;
        self.messages.get_mut(idx)?.tool.as_mut()
    }

    /// The oldest tool block still waiting for `waiting`, if any.
    ///
    /// A turn can emit several calls, and the stream opens a placeholder for
    /// each before any of them commits. The committed call and its result
    /// arrive in the order the calls were made, so they belong to the oldest
    /// block still waiting for them rather than to the newest one.
    fn waiting_tool_block_mut(
        &mut self,
        waiting: impl Fn(&ToolBlock) -> bool,
    ) -> Option<&mut ToolBlock> {
        let idx = self.waiting_tool_block_index(waiting)?;
        self.messages.get_mut(idx)?.tool.as_mut()
    }

    /// Close out placeholders whose call never committed.
    ///
    /// A part opens a placeholder before its call has an id, and the id is
    /// bound when the part ends. If the provider call finishes with a
    /// placeholder still unbound, the call it was opened for was dropped: rig
    /// rejects a call it cannot name, and a rejected call is never routed to
    /// execution. Leaving the block waiting would be worse than untidy, since
    /// part positions restart at zero each turn and the next turn's first part
    /// would bind to it.
    fn close_unbound_placeholders(&mut self) {
        for msg in &mut self.messages {
            let Some(block) = msg.tool.as_mut() else {
                continue;
            };
            if block.call.is_some() {
                continue;
            }
            block.part = None;
            block.streaming = false;
            block.done = true;
            if block.result.is_empty() {
                block.result = "the call was not run".to_string();
            }
        }
    }

    /// Mark every call that was in flight when the run was abandoned.
    ///
    /// A cancelled run surfaces no result for the call it interrupted, so
    /// nothing else will ever fill the block in. Left alone it would read as
    /// still running for the rest of the session, and the user would be
    /// waiting on a call that is never coming back.
    fn mark_in_flight_cancelled(&mut self) {
        for msg in &mut self.messages {
            let Some(block) = msg.tool.as_mut() else {
                continue;
            };
            if block.done {
                continue;
            }
            block.streaming = false;
            block.done = true;
            block.outcome = ToolOutcome::Cancelled;
            if block.result.is_empty() {
                block.result = crate::error::CANCELLED_TEXT.to_string();
            }
        }
    }

    /// Append a tool block, keeping the conversation in the order it happened.
    ///
    /// The assistant message for a run is created up front so streamed text
    /// has somewhere to go, which means a call that arrives before any text
    /// would otherwise be painted *after* the answer it produced. While that
    /// message is still empty the block belongs in front of it.
    fn push_tool_block(&mut self, block: ToolBlock) {
        let before = match self.assistant_idx {
            Some(idx)
                if idx < self.messages.len() && self.messages[idx].content.trim().is_empty() =>
            {
                Some(idx)
            }
            _ => None,
        };
        match before {
            Some(idx) => {
                self.messages.insert(idx, ChatMessage::tool(block));
                self.assistant_idx = Some(idx + 1);
            }
            None => {
                self.messages.push(ChatMessage::tool(block));
            }
        }
    }

    /// Whether the tool block at `idx` is expanded. Collapsed is the default.
    pub fn is_expanded(&self, idx: usize) -> bool {
        self.expanded.contains(&idx)
    }

    /// Expand or collapse the tool block at `idx`.
    pub fn toggle_expanded(&mut self, idx: usize) {
        if !self.expanded.remove(&idx) {
            self.expanded.insert(idx);
        }
    }

    /// Expand or collapse every tool block in the conversation.
    pub fn toggle_all_expanded(&mut self) {
        let tool_idxs: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == Role::Tool)
            .map(|(i, _)| i)
            .collect();
        if tool_idxs.is_empty() {
            return;
        }
        // If anything is still collapsed, expand everything; otherwise fold.
        let any_collapsed = tool_idxs.iter().any(|i| !self.expanded.contains(i));
        for i in tool_idxs {
            if any_collapsed {
                self.expanded.insert(i);
            } else {
                self.expanded.remove(&i);
            }
        }
    }

    /// The index of the most recent tool block, for the keyboard shortcut.
    pub fn last_tool_index(&self) -> Option<usize> {
        self.messages.iter().rposition(|m| m.role == Role::Tool)
    }

    /// The conversation's tool calls, grouped into runs of one category.
    ///
    /// A group ends where the category changes or a non-tool message
    /// interrupts, so the grouping follows the order the calls actually
    /// happened in rather than reordering them by kind.
    pub fn tool_groups(&self) -> Vec<ToolGroup> {
        let mut groups: Vec<ToolGroup> = Vec::new();

        for (i, msg) in self.messages.iter().enumerate() {
            let Some(block) = msg.tool.as_ref() else {
                continue;
            };
            let category = crate::tools::ToolCategory::of(&block.name);

            match groups.last_mut() {
                // Contiguous with the previous call and the same kind of act:
                // the same group.
                Some(g) if g.last + 1 == i && g.category == category => {
                    g.last = i;
                    g.members.push(i);
                }
                _ => groups.push(ToolGroup {
                    first: i,
                    last: i,
                    category,
                    members: vec![i],
                }),
            }
        }

        groups
    }

    /// Whether the group starting at `first` is folded. Open is the default.
    pub fn is_group_folded(&self, first: usize) -> bool {
        self.folded.contains(&first)
    }

    /// Fold or unfold the group starting at `first`.
    pub fn toggle_group(&mut self, first: usize) {
        if !self.folded.remove(&first) {
            self.folded.insert(first);
        }
    }

    /// Fold every group, or unfold every group if any is already folded.
    pub fn toggle_all_groups(&mut self) {
        let firsts: Vec<usize> = self.tool_groups().iter().map(|g| g.first).collect();
        if firsts.is_empty() {
            return;
        }
        let any_open = firsts.iter().any(|f| !self.folded.contains(f));
        for f in firsts {
            if any_open {
                self.folded.insert(f);
            } else {
                self.folded.remove(&f);
            }
        }
    }

    /// The first message index of the most recent group, for the keyboard
    /// shortcut.
    pub fn last_group_first(&self) -> Option<usize> {
        self.tool_groups().last().map(|g| g.first)
    }

    fn append_assistant(&mut self, text: &str) {
        let idx = match self.assistant_idx {
            Some(idx) if idx < self.messages.len() => idx,
            _ => {
                self.messages
                    .push(ChatMessage::new(Role::Assistant, String::new()));
                let idx = self.messages.len() - 1;
                self.assistant_idx = Some(idx);
                idx
            }
        };
        self.messages[idx].content.push_str(text);
    }

    /// Whether the assistant message the current run writes into has no text
    /// yet.
    ///
    /// The renderer asks this to decide whether a header row is the one being
    /// written, so it is `pub` for that. A run whose only output so far is tool
    /// calls has an empty assistant message, and its header is then the only
    /// place the run's state can appear.
    pub fn assistant_is_empty(&self) -> bool {
        match self.assistant_idx.and_then(|i| self.messages.get(i)) {
            Some(msg) => msg.content.trim().is_empty(),
            None => true,
        }
    }

    /// Remove the text streamed for a rejected turn. `turn` is the one-based
    /// model-call index, so the buffer for it is at `turn - 1`.
    fn discard_turn(&mut self, turn: usize) {
        let idx = turn.saturating_sub(1);
        let len = self
            .turn_text
            .get(idx)
            .map(|s| s.chars().count())
            .unwrap_or(0);
        if len > 0 {
            if let Some(msg) = self.assistant_idx.and_then(|i| self.messages.get_mut(i)) {
                let keep = msg.content.chars().count().saturating_sub(len);
                msg.content = msg.content.chars().take(keep).collect();
            }
        }
        if let Some(buf) = self.turn_text.get_mut(idx) {
            buf.clear();
        }
    }

    fn push_activity(&mut self, activity: Activity) {
        self.activity.push(activity);
        // The panel is a timeline of the current run, not an unbounded log.
        if self.activity.len() > 200 {
            self.activity.drain(..self.activity.len() - 200);
        }
    }

    /// Release the bodies of the oldest tool results, keeping their rows.
    ///
    /// The transcript is the user's record of the session, so a row is never
    /// dropped: what the call was, what it was made with, and how it ended
    /// all stay. What goes is the text, which is the part that grows without
    /// bound and the part that is only read again by scrolling back to a row
    /// and expanding it.
    ///
    /// The most recent results are the ones a run is still working from, so
    /// the release starts at the oldest end and stops as soon as the count is
    /// back inside the cap.
    fn release_old_bodies(&mut self) {
        let mut held = self
            .messages
            .iter()
            .filter(|m| m.tool.as_ref().is_some_and(|b| b.done && !b.released))
            .count();
        if held <= MAX_RETAINED_BODIES {
            return;
        }
        for msg in &mut self.messages {
            if held <= MAX_RETAINED_BODIES {
                break;
            }
            let Some(block) = msg.tool.as_mut() else {
                continue;
            };
            if block.released || !block.done {
                continue;
            }
            block.result = RELEASED_BODY.to_string();
            block.released = true;
            held -= 1;
        }
    }

    /// Record a stage transition, if the run has actually moved.
    ///
    /// The stage is derived from the calls that have been made, so it is
    /// reported rather than announced: a run that reads a second file is
    /// still gathering, and saying "gathering" again would fill the timeline
    /// with rows that say nothing. Only a move is worth a row.
    fn emit_stage(&mut self, stage: crate::session::Stage) {
        if !self.session.enter(stage) {
            return;
        }
        self.push_activity(Activity {
            kind: ActivityKind::Stage,
            title: stage.label().to_string(),
            detail: stage.detail().to_string(),
            is_error: false,
            part: None,
        });
    }

    /// Note a call the model made, and move the run's stage if it moved.
    ///
    /// The command is noted here rather than at the result, because a tool
    /// result carries the tool's name but not its arguments: by the time the
    /// result arrives the command is no longer knowable, and a check that
    /// cannot be recognised as a check is a run that reports itself
    /// unverified after passing its tests.
    fn note_call(&mut self, call: &rig::message::ToolCall) {
        let name = call.function.name.to_string();
        if name == "shell" {
            if let Some(command) = call
                .function
                .arguments
                .get("command")
                .and_then(|v| v.as_str())
            {
                self.session.note_command(command);
            }
        }
        // Remember which file a write named, so the result that follows can
        // be attributed to it. The result does not carry the path, and the
        // journal entry the write already made is matched on it.
        if crate::session::is_write_tool(&name) {
            if let Some(path) = call.function.arguments.get("path").and_then(|v| v.as_str()) {
                self.call_paths
                    .insert(call.id.to_string(), path.to_string());
            }
        }
        if let Some(stage) = self.session.note_call(call) {
            self.emit_stage(stage);
        }
        // The correction budget, spent. Past this the model is not converging
        // on the failure, and the honest thing is to stop and report rather
        // than spend the rest of the turn budget on the same error.
        if self.session.corrections_exhausted() {
            self.stop_for_budget(format!(
                "{} corrections have been made without the check passing",
                crate::session::MAX_CORRECTIONS
            ));
        }
        // The same call, once more than the dispatch hook will answer for. The
        // second identical call was skipped and the model told what it already
        // had; a third is the evidence that being told did not land, and the
        // rest of the turn budget would be spent on the same call. This is the
        // guard the failure count cannot be, because a call that succeeds never
        // fails at anything.
        if let Some(call) = self.session.repeated_call() {
            let call = call.to_string();
            self.stop_for_budget(format!(
                "the same call has now been made {} times in a row: {call}",
                crate::session::REPEAT_CALL_STOP_LIMIT
            ));
        }
    }

    /// Note a result, and move the run's stage if it moved.
    fn note_result(&mut self, result: &rig::message::ToolResult) {
        let name = result.name.to_string();
        let call = result.call.to_string();

        // Whether a command passed is the runner's answer, not the result's: a
        // failing `cargo check` is a successful tool call reporting a failed
        // command, so the error flag says nothing about the check. A command
        // that was denied has no outcome at all, which is what keeps a refusal
        // from being counted as a check that failed.
        let outcome = if crate::session::is_command_tool(&name) {
            self.session
                .pending_command()
                .and_then(|command| self.rt.take_outcome(&command))
        } else {
            None
        };

        // The result is noted before the path is filled in: the session records
        // that a change happened, and the app then says which file. The other
        // order leaves the change recorded twice, once with the path and once
        // without.
        if let Some(stage) = self.session.note_result(result, outcome.as_ref()) {
            self.emit_stage(stage);
        }
        if crate::session::is_write_tool(&name) && !result.is_error {
            if let Some(path) = self.call_paths.remove(&call) {
                // The run's own record of what it changed, and the journal's
                // attribution of the action to the call that asked for it.
                // Both are needed: the session answers "has this been
                // checked", the journal answers "what did the run do" after
                // the session has been replaced by the next run's.
                self.session.note_change(&path);
                self.rt.actions().attribute(&call, &path);
            }
        }
        // The same failure, unchanged, twice. The second one is the evidence
        // that the first correction did not address the cause, so a third
        // attempt is a loop rather than a correction.
        if let Some(failure) = self.session.repeated_failure() {
            let failure = failure.to_string();
            self.stop_for_budget(format!(
                "the same failure has now repeated {} times: {failure}",
                crate::session::REPEAT_LIMIT
            ));
        }
    }

    /// Stop the run because a bound was reached, and say which.
    ///
    /// A stop rather than a silent truncation: the model is not asked again,
    /// and the user is told why. The run's outcome is then reported by the
    /// terminal event that follows, so a run stopped here still says what it
    /// changed and whether anything checked it.
    fn stop_for_budget(&mut self, why: String) {
        if !self.streaming || self.cancelling {
            return;
        }
        self.messages.push(ChatMessage::new(
            Role::System,
            format!("Stopping the run: {why}."),
        ));
        self.cancel_run();
    }

    /// What the run in flight has done so far.
    pub fn session(&self) -> &crate::session::Session {
        &self.session
    }

    /// The configuration this session was built with.
    ///
    /// The renderer needs the context window and the compaction threshold to
    /// draw the indicator, and it must read the *resolved* configuration
    /// rather than the environment: a session built with
    /// [`Config::with_window`](crate::config::Config::with_window) has to be
    /// drawn against the window it was actually given.
    pub fn config(&self) -> &crate::config::Config {
        self.rt.config()
    }

    /// Say how the run ended, and what that means for what it changed.
    ///
    /// The one place the outcome is turned into words. A run that changed
    /// nothing says nothing: a line on every answer would be noise, and the
    /// absence of a line is the honest report for a question that needed no
    /// edit. A run that changed something and did not check it says so, and
    /// names the check it should have run, because that is the difference
    /// between "this works" and "this compiles on my machine".
    fn report_outcome(&mut self, terminal: crate::session::Terminal) {
        use crate::session::Outcome;
        let outcome = self.session.outcome(terminal);
        let message = match outcome {
            Outcome::NoChanges => return,
            Outcome::Verified => {
                let checks = self.session.verifications().len();
                let files = self.session.changed().len();
                format!(
                    "Verified: {files} file{} changed, {checks} check{} run after the last \
                     change.",
                    if files == 1 { "" } else { "s" },
                    if checks == 1 { "" } else { "s" },
                )
            }
            Outcome::Unverified { reason } => {
                let files = self.session.changed().len();
                let hint = self
                    .session
                    .verification_hint()
                    .map(|h| format!(" Run `{h}` to check it."))
                    .unwrap_or_default();
                format!(
                    "Unverified: {files} file{} changed, but {reason}.{hint}",
                    if files == 1 { "" } else { "s" },
                )
            }
            Outcome::Cancelled => {
                let files = self.session.changed().len();
                if files == 0 {
                    return;
                }
                format!(
                    "Cancelled after changing {files} file{}. The changes are on disk.",
                    if files == 1 { "" } else { "s" },
                )
            }
            Outcome::Failed(why) => {
                let files = self.session.changed().len();
                if files == 0 {
                    return;
                }
                format!(
                    "Failed after changing {files} file{}: {why}. The changes are on disk.",
                    if files == 1 { "" } else { "s" },
                )
            }
        };
        self.messages.push(ChatMessage::new(Role::System, message));
    }

    /// Move the transcript view by `delta` lines, leaving follow mode.
    ///
    /// Clamped against the height the renderer last reported, so holding
    /// PageDown past the end parks the view on the last row instead of
    /// running the offset up to the `u16` ceiling, where the next PageUp
    /// would appear to do nothing.
    pub fn scroll_by(&mut self, delta: i32) {
        // Clamped from the *current* end, not from the stored offset: the
        // transcript can shrink under the view (`/clear`, a folded group), and
        // a stale offset above the new end would swallow the first presses.
        let max = self.scroll_max.get() as i32;
        let from = (self.scroll as i32).min(max);
        self.scroll = (from + delta).clamp(0, max) as u16;
        self.auto_scroll = false;
    }

    /// Record the transcript's scrollable height, as measured by the renderer.
    pub fn set_scroll_max(&self, max: u16) {
        self.scroll_max.set(max);
    }

    /// The transcript's scrollable height, as of the last frame.
    pub fn scroll_max(&self) -> u16 {
        self.scroll_max.get()
    }

    /// How many message bodies have been rendered since the app started.
    ///
    /// The renderer keeps the lines it produced for each message so that a
    /// redraw does not parse the same markdown again. Whether that is working
    /// is not visible in the painted screen, and a test cannot time a frame
    /// reliably, so the count is what a test asserts on: a redraw of a
    /// conversation that has not changed must not move it.
    pub fn bodies_rendered(&self) -> u64 {
        self.body_cache.borrow().renders()
    }

    /// How many message bodies the renderer is holding.
    pub fn cached_bodies(&self) -> usize {
        self.body_cache.borrow().len()
    }

    pub fn scroll_to_bottom(&mut self) {
        self.auto_scroll = true;
    }
}

/// The message a caught panic carries.
///
/// A panic payload is `Box<dyn Any>` and is usually a `&str` or a `String`.
/// Anything else is described by type rather than by value, because the
/// alternative is a second panic while reporting the first.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "the run's task panicked".to_string()
    }
}

/// A one-line summary of a call's arguments, for the activity timeline and the
/// evals that read it.
fn summarize_args(args: &serde_json::Map<String, serde_json::Value>) -> String {
    if args.is_empty() {
        return String::new();
    }
    args.iter()
        .map(|(k, v)| {
            let value = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            format!("{k}={}", truncate(&value, 60))
        })
        .collect::<Vec<_>>()
        .join("  ")
}

/// Truncate to `max` characters, marking that text was dropped.
fn truncate(text: &str, max: usize) -> String {
    let flat = text.replace('\n', " ");
    if flat.chars().count() <= max {
        return flat;
    }
    let head: String = flat.chars().take(max).collect();
    format!("{head}...")
}

const HELP_TEXT: &str = "\
## Tools

| tool | purpose |
| --- | --- |
| `read_file(path)` | read a file's contents |
| `write_file(path, content)` | write a file |
| `edit_file(path, old, new)` | replace text in a file |
| `list_dir(path)` | list a directory |
| `grep(pattern, path)` | search file contents (regex) |
| `glob(pattern)` | find files by glob pattern |
| `shell(command)` | run a shell command |

## Commands

- `/help` show this help
- `/model` pick a model from Ollama
- `/compact` summarise the conversation to free context
- `/clear` reset the conversation
- `/quit` exit the agent

## Keys

- `Enter` sends, `Shift+Enter` inserts a newline, `Tab` completes a `/command`
- `Up`/`Down` recall previous prompts
- `PageUp`/`PageDown` scroll the transcript, `Ctrl+Up`/`Ctrl+Down` one line
- `Ctrl+Home`/`Ctrl+End` the transcript's top / back to following the tail
- `Left`/`Right` move the caret, `Ctrl+Left`/`Ctrl+Right` by word
- `Home`/`End` the caret's line, `Delete` removes the character under it
- `Ctrl+A`/`Ctrl+E` line start/end, `Ctrl+U`/`Ctrl+K` kill to start/end
- `Ctrl+W` deletes the word before the caret
- `Ctrl+T` fold/unfold the newest tool block, `Ctrl+Shift+T` all of them
- `Ctrl+G` fold/unfold the newest tool group, `Ctrl+Shift+G` all of them
- `Esc` or `Ctrl+C` stop the run in flight; again to quit

## Tool groups

Consecutive tool calls of one kind are grouped under a heading, so a run of
eight reads reads as one `EXPLORE` block rather than as eight unrelated rows.
The heading carries the run's size and how much it returned; folding it
replaces the calls with a digest of which tools ran. Every call is kept: the
grouping organises the transcript, it does not drop anything from it.

| heading | tools |
| --- | --- |
| `EXPLORE` | `read_file`, `list_dir`, `grep`, `glob` |
| `EDIT` | `write_file`, `edit_file` |
| `RUN` | `shell` |

## Environment

- `LLM_PROVIDER` which provider to use: `ollama` (the default), `anthropic`,
  or `openai`. Anything else is an error rather than a silent fallback, and
  the two hosted providers need `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` set.
- `LLM_MODEL` the model to use. `/model` lists what the local daemon has.
- `OLLAMA_BASE_URL` the daemon to talk to (default `http://localhost:11434`).
- `LLM_MAX_TURNS` model calls allowed per prompt (default 100). A tool
  round-trip costs two, so a small budget fails a prompt mid-exploration.
- `LLM_NUM_CTX` context window, in tokens, asked of an Ollama daemon
  (default 262144). The daemon's own default is chosen from the GPU's VRAM
  and is only 4096 on a 6 GB card, which a few tool calls overrun. The window
  is what every other budget is a fraction of, so a small one costs the agent
  its memory of the run; a local model on a small card can be held to a
  smaller window here.
- `LLM_COMPACT_AT` context usage, as a percentage of `LLM_NUM_CTX`, at which
  the conversation is compacted automatically (default 80). `/compact` does
  the same thing on demand.
- `LLM_TOOL_RESULT_CHARS` characters of a single tool result the model may
  see (default: an eighth of `LLM_NUM_CTX`, in tokens, at ~4 chars/token).
  A longer result keeps its head and tail and says how much was elided, so
  one `grep` or `shell` call cannot fill the window on its own.
- `LLM_EVICT_CHARS` characters of tool results one request may carry
  (default: a quarter of `LLM_NUM_CTX`, in tokens, at ~4 chars/token). The
  history is re-sent on every model call, so a run that makes many calls
  grows the prompt by a result per call; once the budget is full the older
  results are replaced by a placeholder that names the tool and says to
  call it again. The result of the call just made is always sent whole.";

//! Events the background agent task sends to the UI loop.
//!
//! The task owns the rig stream and translates every `MultiTurnStreamItem`
//! variant into one of these, so the render loop never has to know about rig
//! types or about the order the stream delivers them in.

use rig::message::{ToolCall, ToolResult};
use rig::run::{CompletionCall, PromptResponse};

/// How many events may be queued for the UI before the producer waits.
///
/// The channel used to be unbounded, which meant a fast model and a slow
/// terminal grew the queue without limit: every streamed fragment was a
/// message, and a long answer is thousands of them. A bound turns that into
/// backpressure, and the coalescing in `session::Emitter` is what keeps the
/// backpressure from being felt as a stutter.
///
/// The approval channel is deliberately *not* bounded the same way. It carries
/// at most one request at a time, because the approver serializes the prompts:
/// a second call cannot ask until the first has been answered.
pub const EVENT_QUEUE: usize = 256;

/// One thing that happened while the agent was running.
#[derive(Debug, Clone)]
pub enum Event {
    /// A live text fragment from the model.
    TextDelta(String),
    /// A live reasoning fragment from the model.
    ReasoningDelta(String),
    /// A tool call's part opened: the model named the tool and is about to
    /// stream its arguments.
    ///
    /// `part` is the part's position in the response, which is the only
    /// identity a call has before it commits: the provider's call id is not
    /// known until the part ends. It is what lets
    /// [`Event::ToolCallIdentified`] bind the placeholder to the call.
    ToolCallStreaming { name: String, part: usize },
    /// A tool call's part ended, naming the call it settled on.
    ///
    /// The call id is what the execution commit and the result are keyed by,
    /// so this is the moment the block opened by
    /// [`Event::ToolCallStreaming`] stops being a placeholder and becomes a
    /// particular call.
    ToolCallIdentified { part: usize, call: String },
    /// A fragment of a tool call's raw argument JSON.
    ///
    /// `part` is the part the fragment belongs to. A turn can stream the
    /// arguments of several calls at once, so the fragment's position in the
    /// response is what keeps it with the call that produced it.
    ToolArgsDelta { part: usize, json: String },
    /// A tool call the model emitted, reported as its turn commits and the
    /// call is about to run.
    ToolCallStarted(Box<ToolCall>),
    /// A tool call Rig actually executed, with any hook rewrite applied.
    ToolExecuted(Box<ToolCall>),
    /// The result of an executed (or hook-skipped) call.
    ToolResult(Box<ToolResult>),
    /// One completion request finished, with its usage and finish reason.
    Completion(Box<CompletionCall>),
    /// A hook rejected a finished turn: text streamed for `turn` was
    /// provisional and must be discarded.
    TurnRetried { turn: usize },
    /// The run finished, carrying the same response `.await` would return.
    Finished(Box<PromptResponse>),
    /// The user interrupted the run and the stream task has stopped. Sent
    /// instead of [`Event::Finished`], so the UI can tell a run that was cut
    /// short from one that completed.
    Cancelled,
    /// The stream ended without a final response.
    ///
    /// A provider that closes the connection mid-run, or a task that is
    /// dropped, leaves the stream with nothing more to yield. Without this
    /// the loop would simply stop reading and the app would sit in
    /// `streaming` forever, with a spinner and no way to type.
    StreamEnded,
    /// The model call failed before it produced anything, and is being tried
    /// again.
    ///
    /// A local daemon routinely drops the very first request while it loads
    /// weights, and the failure is indistinguishable from a model that has
    /// nothing to say. The retry is reported rather than silent: a run that
    /// quietly re-sends a prompt is a run whose timing the user cannot
    /// account for.
    StreamRetry { reason: String },
    /// The task that owns the stream died without sending a terminal event.
    ///
    /// A panic in the task is invisible otherwise: the handle is only joined
    /// on the way out, and by then the app has been waiting on a channel that
    /// will never produce anything again.
    TaskFailed(String),
    /// The `/model` picker's list arrived, or the fetch failed. The error is
    /// carried as text because `Event` is `Clone` and `anyhow::Error` is not.
    ModelsLoaded(Result<Vec<String>, String>),
    /// A compaction run finished: the summary of the conversation, or why it
    /// could not be produced. Carried as text for the same reason as
    /// [`Event::ModelsLoaded`].
    Compacted(Result<String, String>),
    /// The run failed.
    Error(String),
}

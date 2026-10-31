//! What one run did, as it did it.
//!
//! The loop the model is asked to follow is analyze, gather, plan, execute,
//! verify, finish. Until now that loop existed only in the prompt: the app
//! saw a stream of deltas and tool results and had no idea which stage the
//! run was in, whether anything had been changed, or whether anything had
//! been checked. A run that edited three files and ran nothing looked exactly
//! like a run that edited nothing.
//!
//! [`Session`] is the record that makes the difference visible. It is fed
//! every tool call and result as they arrive, and it answers three questions
//! the app could not answer before: which stage is this, what has been
//! changed, and has anything verified it.
//!
//! It is deliberately a plain value with no I/O and no clock. The stage
//! transitions are a function of the calls that have been made, so they can
//! be tested by feeding it calls rather than by running a model.

use rig::message::{AssistantContent, Message as RigMessage, ToolCall, ToolResult};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use crate::event::Event;
use crate::runtime::{CancelToken, CommandOutcome};

/// The stages a run moves through.
///
/// The order is the order of the prompt, and [`Stage::ALL`] is what the
/// prompt test walks: a stage the prompt never names would be a stage the
/// model was never asked to work in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    /// Working out what is being asked.
    Analyze,
    /// Reading the specific things the change needs.
    Gather,
    /// Stating what will be done before doing it.
    Plan,
    /// Changing files.
    Execute,
    /// Running a check on what was changed.
    Verify,
    /// Reporting the result.
    Finish,
}

impl Stage {
    /// Every stage, in order.
    pub const ALL: [Stage; 6] = [
        Stage::Analyze,
        Stage::Gather,
        Stage::Plan,
        Stage::Execute,
        Stage::Verify,
        Stage::Finish,
    ];

    /// The stage's name, as the prompt and the status line spell it.
    pub fn label(self) -> &'static str {
        match self {
            Stage::Analyze => "Analyze",
            Stage::Gather => "Gather",
            Stage::Plan => "Plan",
            Stage::Execute => "Execute",
            Stage::Verify => "Verify",
            Stage::Finish => "Finish",
        }
    }

    /// Whether reaching this stage means the run has changed something.
    pub fn is_editing(self) -> bool {
        matches!(self, Stage::Execute)
    }

    /// What the stage means, in one line, for the activity timeline.
    ///
    /// The label alone ("Verify") says where the run is; this says what that
    /// means it is doing, which is the part a user who has not read the prompt
    /// cannot infer.
    pub fn detail(self) -> &'static str {
        match self {
            Stage::Analyze => "working out what is being asked",
            Stage::Gather => "reading the context the change needs",
            Stage::Plan => "stating what will be done",
            Stage::Execute => "changing files",
            Stage::Verify => "checking what was changed",
            Stage::Finish => "reporting the result",
        }
    }
}

/// One verification command and what it said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    /// The command, as it was run.
    pub command: String,
    /// Whether it succeeded.
    pub passed: bool,
    /// The first line of what it printed, for the transcript.
    pub summary: String,
}

/// One thing that happened, in the order it happened.
///
/// A single ordered log rather than two lists, because the question the
/// outcome turns on is *which came last*: a check that passed before the last
/// edit verified a version of the file that no longer exists.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Step {
    /// A file was created, replaced or patched.
    Changed(String),
    /// A check was run.
    Checked(Verification),
}

/// The last call the model made, and how many times in a row it has made it.
///
/// The fingerprint is the tool's name and a hash of its arguments, which is
/// everything a tool's result is a function of. The arguments are hashed
/// rather than kept because a `write_file` carries a whole file and the
/// fingerprint is compared on every call: deciding whether the model is
/// looping should not cost a comparison of two hundred kilobytes. The label is
/// the same call in a form short enough to name in a message.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LastCall {
    fingerprint: String,
    label: String,
    count: usize,
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Files were changed and the last check, run after the last change,
    /// passed.
    Verified,
    /// Files were changed and nothing checked them afterwards.
    Unverified {
        /// Why, in a form the user can act on.
        reason: String,
    },
    /// Nothing was changed, so there is nothing to verify.
    NoChanges,
    /// The user stopped the run.
    Cancelled,
    /// The run failed.
    Failed(String),
}

/// How a run reached its end, as the app observed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminal {
    /// The model produced a final response.
    Finished,
    /// The user stopped it.
    Cancelled,
    /// It failed.
    Failed(String),
}

/// How many times a run may be told to correct itself.
///
/// A correction is a failed check followed by another change. Three is enough
/// for a compile error, a test failure and the fix for each; past that the
/// model is not converging and the honest thing is to stop and report rather
/// than spend the rest of the turn budget on it.
pub const MAX_CORRECTIONS: usize = 3;

/// How many times the same failure may repeat before the run is stopped.
///
/// Two, because the second identical failure is the evidence that the first
/// correction did not address the cause. A third attempt at the same error is
/// a loop, and the loop is what the budget is there to bound.
pub const REPEAT_LIMIT: usize = 2;

/// How many times the same call may be made in a row before it is answered
/// without being run.
///
/// Two, for the same reason [`REPEAT_LIMIT`] is two. A tool's result is a
/// function of its arguments, so the second identical call returns what the
/// first one returned: running it again spends a model call on an answer the
/// model already has.
///
/// The second call is not fatal. The dispatch hook skips it and tells the
/// model what it already holds, because a model that repeats a call has
/// usually lost track of the result rather than decided to loop, and the
/// cheapest fix is to say so. [`REPEAT_CALL_STOP_LIMIT`] is what happens when
/// saying so does not land.
pub const REPEAT_CALL_LIMIT: usize = 2;

/// How many times the same call may be made in a row before the run is
/// stopped.
///
/// One more than [`REPEAT_CALL_LIMIT`]: the model has been skipped once and
/// told why, so a third identical call is the evidence that being told did not
/// land, and the rest of the turn budget would go on the same call.
///
/// This is the guard the failure count cannot be. A model that repeats a call
/// which *succeeded* never fails at anything, so nothing is counted, and the
/// run spends its whole turn budget re-reading the same file. That is the
/// shape a loop takes when the tools are working.
pub const REPEAT_CALL_STOP_LIMIT: usize = REPEAT_CALL_LIMIT + 1;

/// How much of a call's arguments a message quotes back.
///
/// A `write_file` carries a whole file, and a message that quoted one back at
/// the user would be worse than the loop it is reporting.
const CALL_LABEL_CHARS: usize = 120;

/// The record of one run.
#[derive(Debug, Clone)]
pub struct Session {
    /// The run this record belongs to, so a journal entry can be attributed
    /// to it.
    run: u64,
    stage: Stage,
    steps: Vec<Step>,
    /// How many times a failed check has been followed by another change.
    corrections: usize,
    /// The last failure seen and how many times in a row it has been seen.
    last_failure: Option<(String, usize)>,
    /// The last call seen and how many times in a row it has been seen.
    last_call: Option<LastCall>,
    /// The call ids whose results have already been attributed to the
    /// journal, so a result delivered twice is not recorded twice.
    attributed: Vec<String>,
    /// The command the shell call in flight will run.
    ///
    /// A tool result carries the tool's name but not its arguments, so the
    /// command is only knowable from the call. The app notes it when the call
    /// arrives and reads it back when the result does, which is what lets the
    /// runner's outcome for that command be found.
    pending_command: Option<String>,
}

impl Session {
    /// A record for run `run`, at the first stage.
    pub fn new(run: u64) -> Self {
        Self {
            run,
            stage: Stage::Analyze,
            steps: Vec::new(),
            corrections: 0,
            last_failure: None,
            last_call: None,
            attributed: Vec::new(),
            pending_command: None,
        }
    }

    /// The run this record belongs to.
    pub fn run(&self) -> u64 {
        self.run
    }

    /// The stage the run is in.
    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// The files this run has changed, in the order it changed them.
    pub fn changed(&self) -> Vec<&str> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                Step::Changed(path) => Some(path.as_str()),
                Step::Checked(_) => None,
            })
            .collect()
    }

    /// The checks this run has run, in order.
    pub fn verifications(&self) -> Vec<&Verification> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                Step::Checked(v) => Some(v),
                Step::Changed(_) => None,
            })
            .collect()
    }

    /// How many times a failed check has been followed by another change.
    pub fn corrections(&self) -> usize {
        self.corrections
    }

    /// Whether the run has changed anything.
    pub fn has_changed(&self) -> bool {
        self.steps.iter().any(|s| matches!(s, Step::Changed(_)))
    }

    /// Note a tool call the model made, and return the stage it moved to.
    ///
    /// `None` means the stage did not change. The stage only ever moves
    /// forward: a run that reads a file after editing one is verifying, not
    /// analyzing, and a status line that flickered backwards would be
    /// reporting the model's tool choice rather than the run's progress.
    pub fn note_call(&mut self, call: &ToolCall) -> Option<Stage> {
        let name = call.function.name.to_string();
        // Counted before the stage moves, so a call that repeats is recorded
        // even when it is the call that ends the run.
        self.note_call_repeat(call);
        // A change that follows a failed check is a correction, and counting
        // it here rather than at the result is what makes the count mean "the
        // model tried again", not "the model changed something": a change made
        // before any check is the first attempt, not a correction of one.
        if is_write_tool(&name) && self.last_step_failed() {
            self.corrections += 1;
        }
        let next = if is_write_tool(&name) {
            Stage::Execute
        } else if is_command_tool(&name) {
            match command_of(call) {
                Some(command) if is_verification_command(&command) => Stage::Verify,
                // A command that is not a check is a change: it can write
                // files, install packages, or move things about. Treating it
                // as anything else would let a run that only ever ran shell
                // commands report that it changed nothing.
                _ => Stage::Execute,
            }
        } else if is_read_tool(&name) {
            // A read before anything has been changed is gathering context; a
            // read after a change is part of verifying it.
            if self.has_changed() {
                Stage::Verify
            } else {
                Stage::Gather
            }
        } else {
            self.stage
        };
        self.advance(next)
    }

    /// Note a tool result, and return the stage it moved to.
    ///
    /// `outcome` is the runner's answer for a command that actually ran, and
    /// `None` for everything else, including a command that was denied. The
    /// distinction matters twice over: a failing `cargo check` is a *successful*
    /// tool call reporting a failed command, so the result's error flag cannot
    /// say whether the check passed; and a command that never ran is not a
    /// check at all, so counting it would let a refusal read as a failure.
    ///
    /// A failed check is what makes a correction: the run has to change
    /// something before the failure is addressed. A failure that repeats
    /// unchanged is counted separately, because the second one is the evidence
    /// that the correction did not work.
    pub fn note_result(
        &mut self,
        result: &ToolResult,
        outcome: Option<&CommandOutcome>,
    ) -> Option<Stage> {
        let name = result.name.to_string();

        // A check that ran is recorded whether it passed or failed: a failed
        // check is the most important thing the run knows, and dropping it
        // would leave the outcome claiming nothing was ever run.
        if let Some(outcome) = outcome {
            if is_verification_command(&outcome.command) {
                if outcome.passed {
                    // The failure is answered. A later failure is a new one
                    // rather than a repeat of this, so the count starts over.
                    self.last_failure = None;
                } else {
                    self.note_failure(failure_message(result));
                }
                let verification = Verification {
                    command: outcome.command.clone(),
                    passed: outcome.passed,
                    summary: outcome.summary.clone(),
                };
                self.steps.push(Step::Checked(verification));
                return self.advance(Stage::Verify);
            }
        }

        // A call that failed outright, such as a refused edit or a path that
        // does not exist, is a failure the model has to answer, and it counts
        // the same way. A call that *succeeded* is not evidence that an earlier
        // failure was addressed: the change it made has not been checked yet,
        // and clearing the count here would let a run that alternates a change
        // with the same failing check look like it was converging.
        if result.is_error {
            self.note_failure(failure_message(result));
        }

        if is_write_tool(&name) && !result.is_error {
            // The path is not in the result, so the call that produced it is
            // what names the file. The app fills it in; the change is recorded
            // here first so that a change whose path cannot be attributed is
            // still counted, which is the conservative direction.
            self.steps.push(Step::Changed(String::new()));
            return self.advance(Stage::Execute);
        }

        None
    }

    /// Count `message` as a failure, or as a repeat of the last one.
    fn note_failure(&mut self, message: String) {
        let repeats = match self.last_failure.take() {
            Some((last, count)) if last == message => count + 1,
            _ => 1,
        };
        self.last_failure = Some((message, repeats));
    }

    /// Record that a change was made to `path`.
    ///
    /// Called by the app, which is the only place that knows which call a
    /// result belongs to and therefore which file it touched.
    pub fn note_change(&mut self, path: &str) {
        // The placeholder pushed by `note_result` is filled in rather than
        // duplicated: the result and the call are the same change.
        if let Some(Step::Changed(existing)) = self
            .steps
            .iter_mut()
            .rev()
            .find(|s| matches!(s, Step::Changed(p) if p.is_empty()))
        {
            *existing = path.to_string();
        } else {
            self.steps.push(Step::Changed(path.to_string()));
        }
    }

    /// Note that a failed check was followed by another change.
    pub fn note_correction(&mut self) {
        self.corrections += 1;
    }

    /// Whether the most recent step was a check that failed.
    ///
    /// The question a correction answers: the model changed something *because*
    /// the last thing it heard was that the check failed. A change made after
    /// a check that passed is not a correction, it is new work.
    fn last_step_failed(&self) -> bool {
        matches!(self.steps.last(), Some(Step::Checked(v)) if !v.passed)
    }

    /// The failure that has repeated, if one has.
    ///
    /// `Some` means the same failure has now been seen [`REPEAT_LIMIT`] times
    /// in a row, which is the point at which another attempt is a loop rather
    /// than a correction.
    pub fn repeated_failure(&self) -> Option<&str> {
        match &self.last_failure {
            Some((message, count)) if *count >= REPEAT_LIMIT => Some(message.as_str()),
            _ => None,
        }
    }

    /// The call that has repeated, if one has.
    ///
    /// `Some` means the same call has now been made
    /// [`REPEAT_CALL_STOP_LIMIT`] times in a row. The second one was skipped
    /// and the model told what it already had, so a third is a loop rather
    /// than a retry. The label is the call, short enough to show.
    pub fn repeated_call(&self) -> Option<&str> {
        match &self.last_call {
            Some(last) if last.count >= REPEAT_CALL_STOP_LIMIT => Some(last.label.as_str()),
            _ => None,
        }
    }

    /// Count `call` as a repeat of the last one, or as a new one.
    ///
    /// Consecutive, not cumulative: a model that reads a file, edits it and
    /// reads it again has made the same call twice, and the second read is the
    /// point of the edit rather than a loop. Only a call that follows itself
    /// with nothing in between is one.
    fn note_call_repeat(&mut self, call: &ToolCall) {
        let fingerprint = fingerprint(call);
        let count = match self.last_call.take() {
            Some(last) if last.fingerprint == fingerprint => last.count + 1,
            _ => 1,
        };
        self.last_call = Some(LastCall {
            fingerprint,
            label: call_label(call),
            count,
        });
    }

    /// Whether the run has spent its correction budget.
    pub fn corrections_exhausted(&self) -> bool {
        self.corrections >= MAX_CORRECTIONS
    }

    /// Whether `call` has already been attributed to the journal.
    pub fn is_attributed(&self, call: &str) -> bool {
        self.attributed.iter().any(|id| id == call)
    }

    /// Note that `call`'s result has been attributed to the journal.
    pub fn note_attributed(&mut self, call: &str) {
        self.attributed.push(call.to_string());
    }

    /// The command to propose next, if the run has changed something it has
    /// not checked.
    ///
    /// A suggestion, not an action: every command goes through approval, and
    /// the model is the one that asks. This is what the app shows so the user
    /// can see that the run is not finished.
    ///
    /// `None` in two cases: the run checked its last change, or the change was
    /// not to Rust and no command would settle it. The second case is why this
    /// returns an `Option` rather than a string. The caller renders the hint as
    /// a command to run, and a sentence in that position reads as a command
    /// that does not exist.
    pub fn verification_hint(&self) -> Option<String> {
        let last_change = self
            .steps
            .iter()
            .rposition(|s| matches!(s, Step::Changed(_)))?;
        let last_check = self
            .steps
            .iter()
            .rposition(|s| matches!(s, Step::Checked(_)));
        if last_check.is_some_and(|check| check > last_change) {
            return None;
        }
        self.changed()
            .iter()
            .any(|p| p.ends_with(".rs"))
            .then(|| "cargo check --all-targets".to_string())
    }

    /// How the run ended, given how it reached its end.
    pub fn outcome(&self, terminal: Terminal) -> Outcome {
        match terminal {
            Terminal::Cancelled => Outcome::Cancelled,
            Terminal::Failed(e) => Outcome::Failed(e),
            Terminal::Finished => {
                // The `else` is not dead code: it is the same condition
                // `has_changed` tests, written where the value is needed, so
                // the invariant is carried by the type rather than by a
                // message on a panic that would take the terminal down with it.
                let Some(last_change) = self
                    .steps
                    .iter()
                    .rposition(|s| matches!(s, Step::Changed(_)))
                else {
                    return Outcome::NoChanges;
                };
                let last_check = self
                    .steps
                    .iter()
                    .rposition(|s| matches!(s, Step::Checked(_)));
                match last_check {
                    // A check that ran before the last change verified a
                    // version of the file that no longer exists, so it does
                    // not count. This is the whole reason the log is ordered.
                    Some(check) if check > last_change => match self.steps.get(check) {
                        Some(Step::Checked(v)) if v.passed => Outcome::Verified,
                        _ => Outcome::Unverified {
                            reason: format!("the last check failed: {}", self.last_check_summary()),
                        },
                    },
                    _ => Outcome::Unverified {
                        reason: "no check was run after the last change".to_string(),
                    },
                }
            }
        }
    }

    /// The first line of the most recent check's output.
    fn last_check_summary(&self) -> String {
        self.verifications()
            .last()
            .map(|v| v.summary.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "no output".to_string())
    }

    /// Note the command a shell call is about to run.
    ///
    /// Called by the app when the call arrives, because the result that
    /// follows carries the tool's name but not its arguments.
    pub fn note_command(&mut self, command: &str) {
        self.pending_command = Some(command.to_string());
    }

    /// The command the shell call in flight will run, if one is.
    ///
    /// A tool result carries the tool's name but not its arguments, so the
    /// command is only knowable from the call. The app notes it when the call
    /// arrives and reads it back when the result does, which is what lets the
    /// runner's outcome for that command be found.
    pub fn pending_command(&self) -> Option<String> {
        self.pending_command.clone()
    }

    /// Move to `next`, if it is forward of where the run already is.
    ///
    /// Public because the app announces the stage it is entering, and the
    /// announcement has to be the same decision as the move: two callers each
    /// deciding whether the stage changed would be two answers to one
    /// question. Returns whether it moved.
    pub fn enter(&mut self, next: Stage) -> bool {
        self.advance(next).is_some()
    }

    /// Move to `next`, if it is forward of where the run already is.
    fn advance(&mut self, next: Stage) -> Option<Stage> {
        if next > self.stage {
            self.stage = next;
            Some(next)
        } else {
            None
        }
    }
}

/// How much buffered text is handed over with a wait rather than an offer.
///
/// Only reached when the UI is behind: the ordinary path offers every fragment
/// the moment it arrives, so text appears as it is produced. This is the
/// ceiling on what the producer holds while the queue is full, and reaching it
/// is what converts the producer's own buffer from unbounded growth into
/// backpressure.
const TEXT_COALESCE_LIMIT: usize = 4096;

/// The run's end of the event channel.
///
/// The channel is bounded, so a fragment cannot simply be sent: it may have to
/// wait for room. Waiting on every fragment would make the model's output
/// arrive in lumps, and dropping fragments would lose the answer, so the
/// fragments are *offered* and kept when there is no room. The next fragment
/// is merged with the one being held, which is what makes a slow UI cost
/// fewer, larger messages rather than a growing queue.
///
/// Control events are never coalesced and never dropped: a tool result that
/// went missing would leave the transcript claiming a call never returned.
/// They are sent with a wait, after everything buffered before them, so the
/// order the transcript sees is the order things happened in.
struct Emitter {
    tx: mpsc::Sender<Event>,
    text: String,
    reasoning: String,
    /// Raw argument JSON, one buffer per part.
    ///
    /// A turn can stream several calls at once, and their fragments arrive
    /// interleaved. A single buffer would splice one call's arguments into
    /// another's, so the part is what the buffer is keyed by.
    args: BTreeMap<usize, String>,
}

impl Emitter {
    fn new(tx: mpsc::Sender<Event>) -> Self {
        Self {
            tx,
            text: String::new(),
            reasoning: String::new(),
            args: BTreeMap::new(),
        }
    }

    /// Buffer a text fragment, and hand it over if there is room.
    async fn text(&mut self, fragment: String) {
        self.text.push_str(&fragment);
        if self.text.len() >= TEXT_COALESCE_LIMIT {
            self.flush().await;
        } else {
            self.offer_text();
        }
    }

    /// Buffer a reasoning fragment, and hand it over if there is room.
    async fn reasoning(&mut self, fragment: String) {
        self.reasoning.push_str(&fragment);
        if self.reasoning.len() >= TEXT_COALESCE_LIMIT {
            self.flush().await;
        } else {
            self.offer_reasoning();
        }
    }

    /// Buffer a tool-argument fragment, and hand it over if there is room.
    ///
    /// Coalesced for the same reason as text: the arguments of a large write
    /// are tens of kilobytes, and the stream delivers them a few characters at
    /// a time.
    async fn args(&mut self, part: usize, fragment: String) {
        let buffered = self.args.entry(part).or_default();
        buffered.push_str(&fragment);
        if buffered.len() >= TEXT_COALESCE_LIMIT {
            self.flush().await;
        } else {
            self.offer_args(part);
        }
    }

    /// Send a control event, after everything buffered before it.
    async fn control(&mut self, event: Event) {
        self.flush().await;
        let _ = self.tx.send(event).await;
    }

    /// Hand everything buffered over, waiting for room.
    ///
    /// A closed channel is not an error here: the UI is gone, and there is
    /// nothing left to tell.
    async fn flush(&mut self) {
        if !self.text.is_empty() {
            let text = std::mem::take(&mut self.text);
            let _ = self.tx.send(Event::TextDelta(text)).await;
        }
        if !self.reasoning.is_empty() {
            let text = std::mem::take(&mut self.reasoning);
            let _ = self.tx.send(Event::ReasoningDelta(text)).await;
        }
        for (part, json) in std::mem::take(&mut self.args) {
            let _ = self.tx.send(Event::ToolArgsDelta { part, json }).await;
        }
    }

    /// Offer the buffered text, keeping it if the queue is full.
    fn offer_text(&mut self) {
        if self.text.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.text);
        if let Err(mpsc::error::TrySendError::Full(Event::TextDelta(text))) =
            self.tx.try_send(Event::TextDelta(text))
        {
            self.text = text;
        }
    }

    /// Offer the buffered reasoning, keeping it if the queue is full.
    fn offer_reasoning(&mut self) {
        if self.reasoning.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.reasoning);
        if let Err(mpsc::error::TrySendError::Full(Event::ReasoningDelta(text))) =
            self.tx.try_send(Event::ReasoningDelta(text))
        {
            self.reasoning = text;
        }
    }

    /// Offer one part's buffered arguments, keeping them if the queue is full.
    fn offer_args(&mut self, part: usize) {
        let Some(json) = self.args.remove(&part) else {
            return;
        };
        let event = Event::ToolArgsDelta { part, json };
        if let Err(mpsc::error::TrySendError::Full(Event::ToolArgsDelta { part, json })) =
            self.tx.try_send(event)
        {
            self.args.insert(part, json);
        }
    }
}

/// How long to wait before re-sending a request that failed before it
/// produced anything.
///
/// Long enough for a daemon that is loading weights to finish, short enough
/// that a genuinely broken endpoint is reported while the user is still
/// watching. The retry is not a fix for a bad address: it is for the one
/// failure that resolves itself.
const RETRY_BACKOFF: std::time::Duration = std::time::Duration::from_millis(750);

/// Run one turn of the agent, translating the model's stream into events.
///
/// This is the run's orchestration, and it lives here rather than in the app
/// because it is not presentation: it owns the stream, the stop signal and the
/// transcript, and the app only reads what it produces. Keeping it out of the
/// app is what lets the app be about input and drawing, and it is what makes
/// the loop testable without a terminal.
///
/// The transcript is appended to only on a final response. A run that is
/// cancelled or fails leaves the history as it was, which is why the journal
/// exists: the edits are on disk either way, and the record of them must not
/// depend on the run reaching its end.
pub async fn drive(
    agent: rig::agent::Agent,
    history: Arc<Mutex<Vec<RigMessage>>>,
    prompt: String,
    cancel: Arc<CancelToken>,
    tx: mpsc::Sender<Event>,
    rt: Arc<crate::runtime::Runtime>,
) {
    use futures::StreamExt;
    use rig::agent::MultiTurnStreamItem;
    use rig::streaming::{Item, PartKind, StreamEvent};

    let mut out = Emitter::new(tx);

    // A poisoned lock means some other holder panicked mid-clone. The history
    // is a plain `Vec`, so the data is still sound and the run can proceed;
    // `unwrap` here would instead kill the task and leave the UI waiting on a
    // stream that will never send anything.
    let h = history
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();

    let max_turns = rt.config().limits.max_turns;
    let idle = rt.config().stream_idle_timeout;

    // A local daemon routinely drops the very first request while it loads
    // weights, and the failure looks exactly like a model that has nothing to
    // say. One retry, and only when nothing has happened yet: a run that has
    // already executed a tool has changed the workspace, and replaying it
    // would repeat those changes.
    let mut attempt = 0u32;

    'attempts: loop {
        attempt += 1;

        let mut stream = agent
            .prompt(prompt.as_str())
            .history(h.clone())
            .max_turns(max_turns)
            // rig's own guardrails, which otherwise sit at their defaults of
            // `0` and `None`. The repair hook handles what it can; these bound
            // what it cannot, so a model that keeps emitting unusable calls
            // ends the run with a message instead of looping until the turn
            // budget is gone.
            .max_invalid_tool_call_retries(3)
            .max_consecutive_malformed_tool_calls(Some(3))
            .stream();

        // Whether the run reached its final response. The stream ends after
        // that as a matter of course, and reporting that as "the stream ended
        // without a final response" would turn every successful run into a
        // failure.
        let mut finished = false;

        // How many items this attempt has yielded. A failure before the first
        // one is a connection that never got going, which is the case worth
        // retrying; a failure after it is a run that was already under way.
        let mut items = 0usize;

        loop {
            // Cancellation is polled before the stream, so a stop that arrives
            // while a tool is running is acted on as soon as the task is
            // scheduled rather than after the next item. The idle timer is
            // polled last, so an item that is ready when the timer expires is
            // still delivered: the timeout is for a stream that has stopped
            // talking, not for one that is talking slowly.
            let item = tokio::select! {
                biased;
                _ = cancel.notified() => {
                    if cancel.is_stopped() {
                        // Dropping the stream drops the tool future in flight;
                        // a `shell` child dies with it because the command was
                        // spawned with `kill_on_drop`.
                        drop(stream);
                        out.control(Event::Cancelled).await;
                        return;
                    }
                    // A permit left over from an earlier wake-up: the run was
                    // not stopped, so keep going.
                    continue;
                }
                item = stream.next() => match item {
                    Some(item) => item,
                    // The stream ended without a final response. A provider
                    // that closes the connection mid-run leaves it with nothing
                    // more to yield, and returning here silently would leave
                    // the app in `streaming` with a spinner and no way to type.
                    // Say so, unless the run already finished, in which case
                    // this is just the stream closing behind it.
                    None => {
                        if !finished {
                            out.control(Event::StreamEnded).await;
                        }
                        return;
                    }
                },
                // Nothing has arrived for `idle`. A daemon that has stopped
                // answering looks exactly like one that is thinking, and the
                // user has no way to tell the two apart, so the run is
                // abandoned with a message rather than left on a spinner until
                // they press Ctrl-C. The future is rebuilt on every pass
                // through this loop, so the clock restarts at each item.
                _ = tokio::time::sleep(idle) => {
                    drop(stream);
                    rt.record_stream_timeout();
                    out.control(Event::Error(format!(
                        "the model produced nothing for {}s, so the run was \
                         abandoned. Raise LLM_STREAM_IDLE_TIMEOUT_SECS if the \
                         model is simply slow to start.",
                        idle.as_secs()
                    )))
                    .await;
                    return;
                }
            };

            let item = match item {
                Ok(item) => item,
                Err(e) => {
                    // A failure before the first item is a connection that
                    // never got going, which is what a daemon cold-loading
                    // weights does to the first request. Retry it once.
                    //
                    // `items == 0` is the guard that keeps a run from being
                    // replayed, and it is the right one. A tool call is
                    // emitted as an item before the tool runs, and its result
                    // is another, so a run in which any tool has run has
                    // already yielded at least one item. The session's action
                    // journal would be the wrong test: it accumulates across
                    // runs, so guarding on it would stop the retry from ever
                    // firing again once the session had used a tool once.
                    //
                    // A size rejection is excluded. It arrives the same way,
                    // but re-sending the identical request fails identically,
                    // and the app has a recovery for it that this would
                    // pre-empt: compact the history and retry against a
                    // smaller one.
                    let retryable = items == 0
                        && attempt == 1
                        && !crate::context::is_context_rejection(&e.to_string());
                    if retryable {
                        out.control(Event::StreamRetry {
                            reason: e.to_string(),
                        })
                        .await;
                        tokio::time::sleep(RETRY_BACKOFF).await;
                        continue 'attempts;
                    }
                    out.control(Event::Error(e.to_string())).await;
                    return;
                }
            };

            items += 1;

            match item {
                // Live model output: text, reasoning, and tool-call fragments, as
                // parts start, grow and end.
                MultiTurnStreamItem::StreamAssistantItem(item) => match item {
                    Item::Event(event) => match event {
                        StreamEvent::Start { part, kind, name } => {
                            if kind == PartKind::ToolCall {
                                out.control(Event::ToolCallStreaming {
                                    name: name
                                        .map(|n| n.to_string())
                                        .unwrap_or_else(|| "?".to_string()),
                                    part: part.index(),
                                })
                                .await;
                            }
                        }
                        StreamEvent::Text { text, .. } => {
                            out.text(text).await;
                        }
                        StreamEvent::Reasoning { text, .. } => {
                            out.reasoning(text).await;
                        }
                        StreamEvent::Arguments { part, json } => {
                            out.args(part.index(), json).await;
                        }
                        // The part's finalized content. Text and reasoning already
                        // streamed as fragments. A tool call's end is the first
                        // point its provider id is known, and the id is what the
                        // execution commit and the result are keyed by, so it is
                        // forwarded to bind the placeholder to the call.
                        StreamEvent::End { part, content } => {
                            if let AssistantContent::ToolCall(call) = content {
                                out.control(Event::ToolCallIdentified {
                                    part: part.index(),
                                    call: call.id.to_string(),
                                })
                                .await;
                            }
                        }
                    },
                    // A provider payload the decoder does not model. Dropped, but
                    // not silently: a local server whose wire format rig does not
                    // yet decode shows up here and nowhere else, and a run that
                    // quietly ignores half its stream is impossible to diagnose.
                    Item::Unknown(payload) => {
                        tracing::debug!(?payload, "ignoring an unmodelled stream item");
                    }
                },

                // A call the model emitted, reported as its turn commits.
                MultiTurnStreamItem::ToolCall { tool_call } => {
                    out.control(Event::ToolCallStarted(Box::new(tool_call)))
                        .await;
                }

                // A call Rig executed, with any hook rewrite applied.
                MultiTurnStreamItem::ToolExecutionCommitted { tool_call } => {
                    out.control(Event::ToolExecuted(Box::new(tool_call))).await;
                }

                // The result sent back to the model.
                MultiTurnStreamItem::ToolResult { tool_result } => {
                    out.control(Event::ToolResult(Box::new(tool_result))).await;
                }

                // One model call finished: usage, finish reason, id.
                MultiTurnStreamItem::CompletionCall(call) => {
                    out.control(Event::Completion(Box::new(call))).await;
                }

                // A hook rejected the finished turn: its streamed text was
                // provisional and must be discarded.
                MultiTurnStreamItem::ModelTurnRetried { turn } => {
                    out.control(Event::TurnRetried { turn }).await;
                }

                // The run is done. This is the same response `.await` returns, and
                // the run's transcript to carry forward.
                MultiTurnStreamItem::FinalResponse(response) => {
                    finished = true;
                    if let Ok(mut hist) = history.lock() {
                        // `response.messages` is the run's own transcript,
                        // *excluding* the history it started from, so the two are
                        // appended. Assigning would drop every earlier turn and
                        // leave the model with only the last exchange.
                        hist.extend(response.messages.iter().cloned());
                    }
                    out.control(Event::Finished(Box::new(response))).await;
                }

                // `MultiTurnStreamItem` is `#[non_exhaustive]`: a variant added by
                // a future rig release must not break the build.
                _ => {}
            }
        }
    }
}

/// The command a shell call will run, if it names one.
pub fn command_of(call: &ToolCall) -> Option<String> {
    call.function
        .arguments
        .get("command")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

/// What makes two calls the same call: the tool's name and its arguments.
///
/// The arguments are hashed rather than kept. A `write_file` carries a whole
/// file, and this is computed for every call the model makes, so the guard
/// would otherwise hold a copy of every file the run has written.
fn fingerprint(call: &ToolCall) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (key, value) in &call.function.arguments {
        key.hash(&mut hasher);
        value.hash(&mut hasher);
    }
    format!("{} {:016x}", call.function.name, hasher.finish())
}

/// The call, short enough to name in a message.
///
/// The arguments are rendered by hand rather than through `serde_json`'s
/// `Display`, which builds the whole string first: a `write_file` carries a
/// whole file, and the label exists to be short. An argument that does not fit
/// is left out rather than cutting the label off, so the path of a written file
/// survives the content that precedes it.
fn call_label(call: &ToolCall) -> String {
    let mut label = call.function.name.to_string();
    let mut shown = 0;
    let mut elided = 0;
    for (key, value) in &call.function.arguments {
        let rendered = match value {
            serde_json::Value::String(text) => std::borrow::Cow::Borrowed(text.as_str()),
            other => std::borrow::Cow::Owned(other.to_string()),
        };
        let separator = if shown == 0 { " " } else { ", " };
        let width = label.chars().count()
            + separator.chars().count()
            + key.chars().count()
            + 1
            + rendered.chars().count();
        if width > CALL_LABEL_CHARS {
            elided += 1;
            continue;
        }
        label.push_str(separator);
        label.push_str(key);
        label.push('=');
        label.push_str(&rendered);
        shown += 1;
    }
    if elided > 0 {
        label.push_str(&format!(" (+{elided} more)"));
    }
    label
}

/// Whether `name` is a tool that only reads.
pub fn is_read_tool(name: &str) -> bool {
    matches!(name, "read_file" | "grep" | "glob" | "list_dir")
}

/// Whether `name` is a tool that changes the workspace.
pub fn is_write_tool(name: &str) -> bool {
    matches!(name, "write_file" | "edit_file")
}

/// Whether `name` is the shell.
pub fn is_command_tool(name: &str) -> bool {
    name == "shell"
}

/// Whether `command` is a check rather than a change.
///
/// The list is deliberately short and explicit. A command that is not on it
/// is treated as a change, which is the conservative direction: a run that
/// only ever ran `rm -rf target` must not be able to report that it changed
/// nothing, and a run that ran `cargo test` must be able to report that it
/// verified something.
pub fn is_verification_command(command: &str) -> bool {
    let mut words = command.split_whitespace();
    // Skip leading environment assignments, so `RUST_LOG=debug cargo test` is
    // still recognised as a check.
    let program = loop {
        match words.next() {
            Some(word) if word.contains('=') && !word.starts_with('-') => continue,
            other => break other,
        }
    };
    let Some(program) = program else {
        return false;
    };
    let program = program
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(program)
        .trim_end_matches(".exe");
    let subcommand = words.next().unwrap_or("");
    match program {
        "cargo" => matches!(
            subcommand,
            "check" | "test" | "clippy" | "build" | "fmt" | "nextest" | "doc"
        ),
        "git" => matches!(subcommand, "diff" | "status" | "log" | "show"),
        "go" => matches!(subcommand, "test" | "vet" | "build"),
        "npm" | "pnpm" | "yarn" | "bun" => {
            subcommand == "test" || (subcommand == "run" && words.next() == Some("test"))
        }
        "pytest" | "ruff" | "mypy" | "eslint" | "tsc" | "rustc" | "nextest" => true,
        "python" | "python3" => {
            // `python -m pytest`, and nothing else: `python script.py` is a
            // change, not a check.
            subcommand == "-m" && words.next() == Some("pytest")
        }
        _ => false,
    }
}

/// The text of a tool result, joined.
fn text_of(result: &ToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| c.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The first non-empty line of `text`, trimmed to something a status line can
/// hold.
fn first_line(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    line.chars().take(200).collect()
}

/// The message a failure is identified by.
///
/// The first line, because a compiler error's first line is the error and the
/// rest is the context around it. Two failures that differ only in the line
/// number of the same error are the same failure for the purpose of noticing
/// that a correction did not work.
fn failure_message(result: &ToolResult) -> String {
    first_line(&text_of(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::message::{CallId, Text, ToolFunction, ToolName, ToolResultContent};
    use serde_json::json;

    fn call(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: CallId::from_wire("call_1"),
            function: ToolFunction::new(ToolName::new(name).unwrap(), arguments),
            native: None,
        }
    }

    fn result(name: &str, text: &str, is_error: bool) -> ToolResult {
        ToolResult {
            call: CallId::from_wire("call_1"),
            name: ToolName::new(name).unwrap(),
            content: vec![ToolResultContent::Text(Text::new(text))],
            is_error,
        }
    }

    /// The runner's answer for a command that actually ran.
    fn ran(command: &str, passed: bool, summary: &str) -> CommandOutcome {
        CommandOutcome {
            command: command.to_string(),
            passed,
            summary: summary.to_string(),
        }
    }

    #[test]
    fn a_read_before_a_change_gathers_and_a_read_after_one_verifies() {
        let mut session = Session::new(1);
        assert_eq!(session.stage(), Stage::Analyze);
        assert_eq!(
            session.note_call(&call("read_file", json!({ "path": "a.rs" }))),
            Some(Stage::Gather)
        );
        session.note_result(&result("read_file", "fn main() {}", false), None);
        session.note_call(&call("edit_file", json!({ "path": "a.rs" })));
        session.note_result(&result("edit_file", "patched a.rs", false), None);
        assert_eq!(session.stage(), Stage::Execute);
        assert_eq!(
            session.note_call(&call("read_file", json!({ "path": "a.rs" }))),
            Some(Stage::Verify)
        );
    }

    #[test]
    fn the_stage_never_moves_backwards() {
        let mut session = Session::new(1);
        session.note_call(&call("edit_file", json!({ "path": "a.rs" })));
        session.note_result(&result("edit_file", "patched a.rs", false), None);
        assert_eq!(session.stage(), Stage::Execute);
        // A read after an edit is verification, and a read of a second file
        // does not undo that.
        session.note_call(&call("read_file", json!({ "path": "b.rs" })));
        assert_eq!(session.stage(), Stage::Verify);
        assert_eq!(
            session.note_call(&call("grep", json!({ "pattern": "x" }))),
            None
        );
        assert_eq!(session.stage(), Stage::Verify);
    }

    /// The guard the failure count cannot be: a call that succeeds never fails
    /// at anything, so a model that repeats it is invisible to every other
    /// check the session makes.
    ///
    /// The second identical call is not the stop: the dispatch hook skips it
    /// and tells the model what it already has. The third is, because by then
    /// being told has not landed.
    #[test]
    fn the_same_call_three_times_in_a_row_is_a_loop() {
        let mut session = Session::new(1);
        let read = call("read_file", json!({ "path": "a.rs" }));
        session.note_call(&read);
        session.note_result(&result("read_file", "fn main() {}", false), None);
        assert_eq!(session.repeated_call(), None, "one call is not a loop");

        session.note_call(&read);
        assert_eq!(
            session.repeated_call(),
            None,
            "the second is answered by the hook, not stopped"
        );

        session.note_call(&read);
        let label = session
            .repeated_call()
            .expect("a third identical call is a loop");
        assert!(label.contains("read_file"), "{label}");
        assert!(label.contains("a.rs"), "{label}");
    }

    #[test]
    fn a_call_with_different_arguments_is_not_a_repeat() {
        let mut session = Session::new(1);
        session.note_call(&call("read_file", json!({ "path": "a.rs" })));
        session.note_call(&call("read_file", json!({ "path": "b.rs" })));
        assert_eq!(session.repeated_call(), None);
    }

    #[test]
    fn a_repeat_that_is_not_consecutive_is_not_a_loop() {
        let mut session = Session::new(1);
        let read = call("read_file", json!({ "path": "a.rs" }));
        session.note_call(&read);
        session.note_call(&call("edit_file", json!({ "path": "a.rs" })));
        // Reading the file again after editing it is the point of the edit,
        // not a loop.
        session.note_call(&read);
        assert_eq!(session.repeated_call(), None);
    }

    #[test]
    fn a_different_call_clears_the_repeat() {
        let mut session = Session::new(1);
        let read = call("read_file", json!({ "path": "a.rs" }));
        session.note_call(&read);
        session.note_call(&read);
        session.note_call(&read);
        assert!(session.repeated_call().is_some());

        session.note_call(&call("list_dir", json!({ "path": "." })));
        assert_eq!(session.repeated_call(), None);
    }

    /// A `write_file` carries a whole file, and the label is what the stop
    /// message shows. Quoting the file back would be worse than the loop, and
    /// cutting the label off at the budget would lose the path that says which
    /// file it was.
    #[test]
    fn the_repeat_names_the_call_short_enough_to_show() {
        let mut session = Session::new(1);
        let big = call(
            "write_file",
            json!({ "path": "a.rs", "content": "x".repeat(5_000) }),
        );
        session.note_call(&big);
        session.note_call(&big);
        session.note_call(&big);
        let label = session.repeated_call().expect("the call repeated");
        assert!(label.starts_with("write_file "), "{label}");
        assert!(label.contains("path=a.rs"), "{label}");
        assert!(label.contains("+1 more"), "{label}");
        assert!(
            label.chars().count() <= CALL_LABEL_CHARS + 16,
            "{} chars: {label}",
            label.chars().count()
        );
    }

    #[test]
    fn a_check_is_recognised_and_a_change_is_not() {
        assert!(is_verification_command("cargo check --all-targets"));
        assert!(is_verification_command("cargo test --locked"));
        assert!(is_verification_command("RUST_LOG=debug cargo test"));
        assert!(is_verification_command("git diff --stat"));
        assert!(is_verification_command("python -m pytest tests/"));
        assert!(is_verification_command("C:\\tools\\cargo.exe clippy"));
        assert!(!is_verification_command("cargo run"));
        assert!(!is_verification_command("rm -rf target"));
        assert!(!is_verification_command("python script.py"));
        assert!(!is_verification_command("git commit -m x"));
        assert!(!is_verification_command(""));
    }

    #[test]
    fn a_shell_command_that_is_not_a_check_moves_the_run_to_execute() {
        let mut session = Session::new(1);
        session.note_call(&call("shell", json!({ "command": "rm -rf target" })));
        assert_eq!(session.stage(), Stage::Execute);
        // But it is not a *file* change: the command's effect is not knowable
        // from its name, and counting every `ls` as a change would make every
        // run demand a verification it does not need.
        assert!(!session.has_changed());
    }

    #[test]
    fn a_check_that_ran_after_the_last_change_verifies_the_run() {
        let mut session = Session::new(1);
        session.note_call(&call("edit_file", json!({ "path": "a.rs" })));
        session.note_result(&result("edit_file", "patched a.rs", false), None);
        session.note_change("a.rs");
        session.note_command("cargo check");
        session.note_call(&call("shell", json!({ "command": "cargo check" })));
        session.note_result(
            &result("shell", "Finished dev profile", false),
            Some(&ran("cargo check", true, "Finished dev profile")),
        );
        assert_eq!(session.outcome(Terminal::Finished), Outcome::Verified);
    }

    #[test]
    fn a_check_that_ran_before_the_last_change_does_not_verify_it() {
        let mut session = Session::new(1);
        session.note_call(&call("edit_file", json!({ "path": "a.rs" })));
        session.note_result(&result("edit_file", "patched a.rs", false), None);
        session.note_change("a.rs");
        session.note_command("cargo check");
        session.note_call(&call("shell", json!({ "command": "cargo check" })));
        session.note_result(
            &result("shell", "Finished dev profile", false),
            Some(&ran("cargo check", true, "Finished dev profile")),
        );
        // A second edit after the check: the check verified a version of the
        // file that no longer exists.
        session.note_call(&call("edit_file", json!({ "path": "a.rs" })));
        session.note_result(&result("edit_file", "patched a.rs", false), None);
        session.note_change("a.rs");
        assert_eq!(
            session.outcome(Terminal::Finished),
            Outcome::Unverified {
                reason: "no check was run after the last change".to_string()
            }
        );
    }

    #[test]
    fn a_command_that_never_ran_is_not_a_check() {
        // A denied command has no outcome, and a check that never ran must not
        // be recorded: counting it would let a refusal read as a failure, and
        // would let a run claim it verified something on the strength of a
        // command the user stopped.
        let mut session = Session::new(1);
        session.note_change("a.rs");
        session.note_command("cargo check");
        session.note_call(&call("shell", json!({ "command": "cargo check" })));
        session.note_result(&result("shell", "denied: not now", true), None);
        assert!(session.verifications().is_empty());
        assert_eq!(
            session.outcome(Terminal::Finished),
            Outcome::Unverified {
                reason: "no check was run after the last change".to_string()
            }
        );
    }

    #[test]
    fn a_run_that_changed_nothing_needs_no_verification() {
        let mut session = Session::new(1);
        session.note_call(&call("read_file", json!({ "path": "a.rs" })));
        session.note_result(&result("read_file", "fn main() {}", false), None);
        assert_eq!(session.outcome(Terminal::Finished), Outcome::NoChanges);
    }

    #[test]
    fn a_failed_check_is_reported_as_the_reason() {
        let mut session = Session::new(1);
        session.note_call(&call("edit_file", json!({ "path": "a.rs" })));
        session.note_result(&result("edit_file", "patched a.rs", false), None);
        session.note_change("a.rs");
        session.note_command("cargo check");
        session.note_call(&call("shell", json!({ "command": "cargo check" })));
        // A failing `cargo check` is a *successful* tool call: the runner
        // reports the exit code in the output and the tool returns `Ok`. The
        // outcome is what says the check failed.
        session.note_result(
            &result("shell", "error[E0308]: mismatched types", false),
            Some(&ran("cargo check", false, "error[E0308]: mismatched types")),
        );
        assert_eq!(
            session.outcome(Terminal::Finished),
            Outcome::Unverified {
                reason: "the last check failed: error[E0308]: mismatched types".to_string()
            }
        );
    }

    #[test]
    fn the_same_failure_twice_is_a_loop() {
        let mut session = Session::new(1);
        session.note_result(
            &result("shell", "error[E0308]: mismatched types", true),
            None,
        );
        assert_eq!(session.repeated_failure(), None);
        session.note_result(
            &result("shell", "error[E0308]: mismatched types", true),
            None,
        );
        assert_eq!(
            session.repeated_failure(),
            Some("error[E0308]: mismatched types")
        );
    }

    #[test]
    fn a_different_failure_resets_the_count() {
        let mut session = Session::new(1);
        session.note_result(
            &result("shell", "error[E0308]: mismatched types", true),
            None,
        );
        session.note_result(
            &result("shell", "error[E0432]: unresolved import", true),
            None,
        );
        assert_eq!(session.repeated_failure(), None);
    }

    #[test]
    fn a_success_clears_the_failure() {
        let mut session = Session::new(1);
        session.note_result(
            &result("shell", "error[E0308]: mismatched types", true),
            None,
        );
        session.note_result(
            &result("shell", "Finished dev profile", false),
            Some(&ran("cargo check", true, "Finished dev profile")),
        );
        session.note_result(
            &result("shell", "error[E0308]: mismatched types", true),
            None,
        );
        assert_eq!(session.repeated_failure(), None);
    }

    #[test]
    fn a_change_between_two_identical_failures_does_not_reset_the_count() {
        // The loop the count exists to catch: check, change, check again with
        // the same error. The change in between is the correction, and the
        // second failure is the evidence that it did not work. Clearing the
        // count on the change would make the loop invisible.
        let mut session = Session::new(1);
        session.note_result(
            &result("shell", "error[E0308]: mismatched types", false),
            Some(&ran("cargo check", false, "error[E0308]: mismatched types")),
        );
        session.note_result(&result("edit_file", "patched a.rs", false), None);
        session.note_result(
            &result("shell", "error[E0308]: mismatched types", false),
            Some(&ran("cargo check", false, "error[E0308]: mismatched types")),
        );
        assert_eq!(
            session.repeated_failure(),
            Some("error[E0308]: mismatched types")
        );
    }

    #[test]
    fn the_correction_budget_is_bounded() {
        let mut session = Session::new(1);
        for _ in 0..MAX_CORRECTIONS {
            assert!(!session.corrections_exhausted());
            session.note_correction();
        }
        assert!(session.corrections_exhausted());
    }

    #[test]
    fn the_hint_names_cargo_check_for_a_rust_change() {
        let mut session = Session::new(1);
        session.note_change("src/main.rs");
        assert_eq!(
            session.verification_hint().as_deref(),
            Some("cargo check --all-targets")
        );
    }

    #[test]
    fn a_change_with_no_command_to_check_it_gets_no_hint() {
        // The hint is rendered as a command to run, so a change no command
        // would settle has to produce nothing rather than a sentence in that
        // position. The report still says the change is unverified; what it
        // does not do is invent a command.
        let mut session = Session::new(1);
        session.note_change("notes.txt");
        assert_eq!(session.verification_hint(), None);
    }

    #[test]
    fn the_hint_is_gone_once_a_check_has_run() {
        let mut session = Session::new(1);
        session.note_change("src/main.rs");
        session.note_command("cargo check");
        session.note_call(&call("shell", json!({ "command": "cargo check" })));
        session.note_result(
            &result("shell", "Finished dev profile", false),
            Some(&ran("cargo check", true, "Finished dev profile")),
        );
        assert_eq!(session.verification_hint(), None);
    }

    #[test]
    fn a_cancelled_run_is_not_reported_as_unverified() {
        let mut session = Session::new(1);
        session.note_change("src/main.rs");
        assert_eq!(session.outcome(Terminal::Cancelled), Outcome::Cancelled);
    }

    #[test]
    fn a_failed_run_carries_its_error() {
        let session = Session::new(1);
        assert_eq!(
            session.outcome(Terminal::Failed("boom".to_string())),
            Outcome::Failed("boom".to_string())
        );
    }

    #[test]
    fn a_result_is_attributed_once() {
        let mut session = Session::new(1);
        assert!(!session.is_attributed("call_1"));
        session.note_attributed("call_1");
        assert!(session.is_attributed("call_1"));
    }
}

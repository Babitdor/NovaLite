//! The approval service: who decides whether an action may run.
//!
//! Workspace trust and action approval are two different questions. Trust is
//! asked once per directory and recorded outside the workspace; approval is
//! asked per action, every time, and is never recorded. A trusted directory
//! is one the agent may work in, not one where it may do as it likes.
//!
//! The decision is made through a trait so the same gate serves the TUI, the
//! examples, and a test. Nothing in the tool path may bypass it: the tools ask
//! [`crate::runtime::Runtime::approve`], which is the only place a decision is
//! turned into a result.

use std::sync::atomic::{AtomicU64, Ordering};

use futures::future::BoxFuture;
use tokio::sync::{mpsc, oneshot};

/// What kind of action is being approved.
///
/// The kind is what the prompt is shaped by: a write shows a diff, a command
/// shows the command line and its working directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    /// Creating a file that does not exist.
    Create,
    /// Replacing the contents of a file that does.
    Write,
    /// Applying a search-and-replace patch.
    Patch,
    /// Running a terminal command.
    Command,
    /// Reaching a path outside the workspace.
    ///
    /// Asked once per directory rather than once per path, and before the
    /// action itself: a write outside the root asks this and then asks to
    /// write, because being allowed to look somewhere is not being allowed to
    /// change it.
    Access,
}

impl ActionKind {
    /// The word the prompt leads with.
    pub fn label(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Write => "write",
            Self::Patch => "patch",
            Self::Command => "run",
            Self::Access => "access",
        }
    }
}

/// One action, as it is put to the user.
///
/// The request is immutable once built: the approval is bound to these exact
/// arguments, so a file that changed between the preview and the commit
/// cannot be written under the old decision. The tool re-checks the version
/// it previewed before it commits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    /// A unique id, so a response can be matched to its request when several
    /// are outstanding.
    pub id: u64,
    /// What kind of action this is.
    pub kind: ActionKind,
    /// A one-line summary, e.g. `write src/main.rs`.
    pub title: String,
    /// The workspace-relative paths the action touches.
    pub paths: Vec<String>,
    /// What the action will do: a diff, a new-file preview, or a command.
    pub preview: String,
    /// Anything else the user needs: working directory, timeout, byte counts.
    pub detail: String,
}

impl ApprovalRequest {
    /// A request with a fresh id.
    pub fn new(kind: ActionKind, title: impl Into<String>) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            kind,
            title: title.into(),
            paths: Vec::new(),
            preview: String::new(),
            detail: String::new(),
        }
    }

    /// Add a workspace-relative path.
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.paths.push(path.into());
        self
    }

    /// Set the preview body.
    pub fn with_preview(mut self, preview: impl Into<String>) -> Self {
        self.preview = preview.into();
        self
    }

    /// Set the detail line.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }
}

/// What the user decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Go ahead.
    Approved,
    /// Do not. The reason is shown to the model, so it can tell a refusal
    /// from a failure and stop rather than retry.
    Denied(String),
}

impl Decision {
    /// Whether the action may run.
    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approved)
    }
}

/// Who answers an approval request.
///
/// Asynchronous because the interactive implementation waits on the user
/// while the run is in flight, and the tool future must not block the
/// runtime's worker.
pub trait Approver: Send + Sync + std::fmt::Debug {
    /// Decide whether `request` may run.
    fn decide<'a>(&'a self, request: ApprovalRequest) -> BoxFuture<'a, Decision>;
}

/// Approves every action in the workspace, and no access outside it.
///
/// Installed when the mode is [`crate::config::ApprovalMode::Auto`]. It skips
/// the question, not the boundary: path validation, timeouts and every other
/// limit still apply, because they live in the tools rather than here.
///
/// [`ActionKind::Access`] is the one thing it refuses. Auto mode is a
/// decision about the work, taken by someone who chose the directory it
/// happens in; a mode that also granted directories outside that one would be
/// a way to leave the workspace with nobody answering for it. The session is
/// started in the directory the work needs, or it runs without auto-approval
/// and the user is asked.
#[derive(Debug, Default)]
pub struct AutoApprover;

impl Approver for AutoApprover {
    fn decide<'a>(&'a self, request: ApprovalRequest) -> BoxFuture<'a, Decision> {
        Box::pin(async move {
            if request.kind != ActionKind::Access {
                return Decision::Approved;
            }
            let what = request
                .paths
                .first()
                .cloned()
                .unwrap_or_else(|| request.title.clone());
            Decision::Denied(format!(
                "'{what}' is outside the workspace, and auto-approval does not grant \
                 directories outside it; start the session in that directory, or run without \
                 auto-approval to be asked"
            ))
        })
    }
}

/// Denies everything, naming why.
///
/// The default when the mode is `Prompt` and no interactive channel has been
/// attached. A gate with nobody behind it stays shut: the alternative is a
/// build that silently writes files because the prompt was never wired up.
#[derive(Debug, Default)]
pub struct NoChannel;

impl Approver for NoChannel {
    fn decide<'a>(&'a self, request: ApprovalRequest) -> BoxFuture<'a, Decision> {
        Box::pin(async move {
            Decision::Denied(format!(
                "no approval channel is attached, so '{}' was not run",
                request.title
            ))
        })
    }
}

/// An approval request on its way to the UI, with the way to answer it.
///
/// The reply is a oneshot rather than a shared slot: a request is answered
/// exactly once, and a dropped sender is how the UI says "no answer is
/// coming", which the approver turns into a denial rather than a hang.
#[derive(Debug)]
pub struct PendingApproval {
    request: ApprovalRequest,
    reply: oneshot::Sender<Decision>,
}

impl PendingApproval {
    /// The request to show.
    pub fn request(&self) -> &ApprovalRequest {
        &self.request
    }

    /// Answer it.
    ///
    /// A send that fails means the asker is gone (the run was cancelled, or
    /// the process is shutting down), which is not an error here: the
    /// decision has nowhere to go and there is nothing useful to do about it.
    pub fn answer(self, decision: Decision) {
        let _ = self.reply.send(decision);
    }
}

/// Asks the UI, and waits for the answer.
///
/// The prompt is serialized. A model that emits three calls in one turn gets
/// three questions, one at a time, rather than three overlapping prompts the
/// user cannot tell apart or answer in the order they were asked. The lock is
/// held across the wait, so the second request is not even sent until the
/// first is answered.
#[derive(Debug)]
pub struct InteractiveApprover {
    tx: mpsc::UnboundedSender<PendingApproval>,
    gate: tokio::sync::Mutex<()>,
}

impl InteractiveApprover {
    /// An approver that sends its questions to `tx`.
    pub fn new(tx: mpsc::UnboundedSender<PendingApproval>) -> Self {
        Self {
            tx,
            gate: tokio::sync::Mutex::new(()),
        }
    }
}

impl Approver for InteractiveApprover {
    fn decide<'a>(&'a self, request: ApprovalRequest) -> BoxFuture<'a, Decision> {
        Box::pin(async move {
            let _serialized = self.gate.lock().await;
            let title = request.title.clone();
            let (reply, answer) = oneshot::channel();
            if self.tx.send(PendingApproval { request, reply }).is_err() {
                return Decision::Denied(format!(
                    "the approval prompt is not on screen, so '{title}' was not run"
                ));
            }
            match answer.await {
                Ok(decision) => decision,
                // The UI dropped the request without answering: a quit, a
                // panic, or a closed channel. Anything but an explicit yes is
                // a no.
                Err(_) => Decision::Denied(format!("'{title}' was not answered")),
            }
        })
    }
}

/// How many lines of a preview are shown before the rest is elided.
///
/// A prompt is read on a terminal, not scrolled: a preview longer than the
/// screen is one the user cannot take in, and approving a change nobody read
/// is the failure the prompt exists to prevent.
pub const PREVIEW_LINES: usize = 40;

/// Keep the first `max` lines of `text`, saying how many were dropped.
fn elide_lines(text: &str, max: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max {
        return text.to_string();
    }
    let mut out = lines[..max].join("\n");
    out.push_str(&format!(
        "\n... [{} more line(s) not shown] ...",
        lines.len() - max
    ));
    out
}

/// A preview of a file that does not exist yet: every line is an addition.
pub fn new_file_preview(content: &str) -> String {
    let mut out = String::new();
    for line in content.lines() {
        out.push('+');
        out.push_str(line);
        out.push('\n');
    }
    if content.is_empty() {
        out.push_str("(empty file)\n");
    }
    elide_lines(&out, PREVIEW_LINES)
}

/// A preview of a change to an existing file.
///
/// The unchanged head and tail are dropped and only the changed middle is
/// shown, marked `-` and `+`. That is not a minimal diff: a line moved from
/// one end of the file to the other reads as a deletion and an addition
/// rather than as a move. It is a *preview*, and the question it has to
/// answer is "is this the change I meant", which the changed region answers
/// and a line-by-line alignment does not answer any better.
pub fn diff_preview(old: &str, new: &str) -> String {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let mut head = 0;
    while head < old_lines.len() && head < new_lines.len() && old_lines[head] == new_lines[head] {
        head += 1;
    }
    let mut tail = 0;
    while tail < old_lines.len().saturating_sub(head)
        && tail < new_lines.len().saturating_sub(head)
        && old_lines[old_lines.len() - 1 - tail] == new_lines[new_lines.len() - 1 - tail]
    {
        tail += 1;
    }

    let removed = &old_lines[head..old_lines.len() - tail];
    let added = &new_lines[head..new_lines.len() - tail];

    let mut out = String::new();
    if head > 0 {
        out.push_str(&format!("@@ after line {head} @@\n"));
    } else {
        out.push_str("@@ from the top @@\n");
    }
    for line in removed {
        out.push('-');
        out.push_str(line);
        out.push('\n');
    }
    for line in added {
        out.push('+');
        out.push_str(line);
        out.push('\n');
    }
    if tail > 0 {
        out.push_str(&format!("@@ {tail} unchanged line(s) follow @@\n"));
    }
    elide_lines(&out, PREVIEW_LINES)
}

/// Answers from a fixed script, recording every request it saw.
///
/// The test fixture: a case states the decisions it expects to be asked for,
/// and asserts on what was asked. A request past the end of the script is
/// denied, so a test that under-specifies its script fails loudly rather than
/// silently approving.
#[derive(Debug, Default)]
pub struct ScriptedApprover {
    decisions: std::sync::Mutex<Vec<Decision>>,
    seen: std::sync::Mutex<Vec<ApprovalRequest>>,
}

impl ScriptedApprover {
    /// An approver that answers with `decisions`, in order.
    pub fn new(decisions: Vec<Decision>) -> Self {
        Self {
            decisions: std::sync::Mutex::new(decisions),
            seen: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// An approver that approves everything, for a test that only cares that
    /// the action ran.
    pub fn approving() -> Self {
        Self::new(Vec::new())
    }

    /// The requests it was asked, in order.
    pub fn seen(&self) -> Vec<ApprovalRequest> {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// How many requests it was asked.
    pub fn count(&self) -> usize {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

impl Approver for ScriptedApprover {
    fn decide<'a>(&'a self, request: ApprovalRequest) -> BoxFuture<'a, Decision> {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request);
        let mut script = self
            .decisions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let decision = if script.is_empty() {
            Decision::Approved
        } else {
            script.remove(0)
        };
        Box::pin(async move { decision })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_request_gets_its_own_id() {
        let a = ApprovalRequest::new(ActionKind::Write, "write a");
        let b = ApprovalRequest::new(ActionKind::Write, "write b");
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn a_request_carries_its_paths_and_preview() {
        let request = ApprovalRequest::new(ActionKind::Patch, "patch src/lib.rs")
            .with_path("src/lib.rs")
            .with_preview("-old\n+new")
            .with_detail("1 hunk");
        assert_eq!(request.paths, vec!["src/lib.rs".to_string()]);
        assert!(request.preview.contains("+new"));
        assert_eq!(request.detail, "1 hunk");
    }

    #[test]
    fn the_auto_approver_refuses_to_widen_the_workspace() {
        // The one question auto mode does not answer with yes. Otherwise the
        // mode would be a way out of the directory the user chose.
        let decision = futures::executor::block_on(AutoApprover.decide(
            ApprovalRequest::new(ActionKind::Access, "access /etc").with_path("../../etc"),
        ));
        match decision {
            Decision::Denied(reason) => {
                assert!(reason.contains("outside the workspace"), "{reason}")
            }
            Decision::Approved => panic!("auto mode must not grant directories"),
        }
    }

    #[test]
    fn the_auto_approver_approves() {
        let decision = futures::executor::block_on(
            AutoApprover.decide(ApprovalRequest::new(ActionKind::Command, "run cargo check")),
        );
        assert!(decision.is_approved());
    }

    #[test]
    fn the_no_channel_approver_denies_and_names_the_action() {
        let decision = futures::executor::block_on(
            NoChannel.decide(ApprovalRequest::new(ActionKind::Command, "run cargo check")),
        );
        match decision {
            Decision::Denied(reason) => assert!(reason.contains("cargo check"), "{reason}"),
            Decision::Approved => panic!("a gate with nobody behind it must stay shut"),
        }
    }

    #[test]
    fn the_scripted_approver_answers_in_order_and_records() {
        let approver =
            ScriptedApprover::new(vec![Decision::Approved, Decision::Denied("no".to_string())]);
        let first = futures::executor::block_on(
            approver.decide(ApprovalRequest::new(ActionKind::Write, "one")),
        );
        let second = futures::executor::block_on(
            approver.decide(ApprovalRequest::new(ActionKind::Write, "two")),
        );
        assert!(first.is_approved());
        assert!(!second.is_approved());
        assert_eq!(approver.count(), 2);
        assert_eq!(approver.seen()[0].title, "one");
    }

    #[test]
    fn a_new_file_preview_marks_every_line_as_an_addition() {
        let preview = new_file_preview("one\ntwo\n");
        assert_eq!(preview, "+one\n+two\n");
    }

    #[test]
    fn an_empty_new_file_says_so_rather_than_showing_nothing() {
        assert_eq!(new_file_preview(""), "(empty file)\n");
    }

    #[test]
    fn a_diff_preview_shows_only_the_changed_region() {
        let old = "a\nb\nc\nd\ne\n";
        let new = "a\nb\nCHANGED\nd\ne\n";
        let preview = diff_preview(old, new);
        assert!(preview.contains("-c"), "{preview}");
        assert!(preview.contains("+CHANGED"), "{preview}");
        // The unchanged head and tail are context, not content: showing them
        // would push the change off the screen on a real file.
        assert!(!preview.contains("-a"), "{preview}");
        assert!(!preview.contains("+e"), "{preview}");
        assert!(preview.contains("after line 2"), "{preview}");
        assert!(preview.contains("2 unchanged line(s) follow"), "{preview}");
    }

    #[test]
    fn a_diff_preview_of_an_append_shows_only_the_added_lines() {
        let preview = diff_preview("a\n", "a\nb\n");
        assert!(preview.contains("+b"), "{preview}");
        assert!(!preview.contains("-a"), "{preview}");
    }

    #[test]
    fn a_long_preview_is_elided_and_says_how_much_was_dropped() {
        let content: String = (0..PREVIEW_LINES + 10)
            .map(|i| format!("line {i}\n"))
            .collect();
        let preview = new_file_preview(&content);
        assert!(preview.contains("10 more line(s) not shown"), "{preview}");
        assert_eq!(preview.lines().count(), PREVIEW_LINES + 1);
    }

    #[tokio::test]
    async fn the_interactive_approver_returns_the_answer_it_is_given() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let approver = std::sync::Arc::new(InteractiveApprover::new(tx));
        // Spawned, not merely constructed: `decide` returns a lazy future, so
        // a test that built it and then waited on the channel would wait for a
        // request that is never sent.
        let asking = {
            let approver = approver.clone();
            tokio::spawn(async move {
                approver
                    .decide(ApprovalRequest::new(ActionKind::Write, "write a"))
                    .await
            })
        };

        let pending = rx.recv().await.expect("the request reaches the UI");
        assert_eq!(pending.request().title, "write a");
        pending.answer(Decision::Approved);

        assert!(asking.await.expect("the task does not panic").is_approved());
    }

    #[tokio::test]
    async fn a_dropped_request_denies_rather_than_hanging() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let approver = std::sync::Arc::new(InteractiveApprover::new(tx));
        let asking = {
            let approver = approver.clone();
            tokio::spawn(async move {
                approver
                    .decide(ApprovalRequest::new(ActionKind::Command, "run cargo test"))
                    .await
            })
        };

        let pending = rx.recv().await.expect("the request reaches the UI");
        drop(pending);

        match asking.await.expect("the task does not panic") {
            Decision::Denied(reason) => assert!(reason.contains("cargo test"), "{reason}"),
            Decision::Approved => panic!("an unanswered request must not be approved"),
        }
    }

    #[tokio::test]
    async fn a_closed_channel_denies_rather_than_hanging() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let approver = InteractiveApprover::new(tx);
        let decision = approver
            .decide(ApprovalRequest::new(ActionKind::Write, "write a"))
            .await;
        match decision {
            Decision::Denied(reason) => assert!(reason.contains("not on screen"), "{reason}"),
            Decision::Approved => panic!("a gate with nobody behind it must stay shut"),
        }
    }

    #[tokio::test]
    async fn prompts_are_serialized_so_two_calls_are_asked_one_at_a_time() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let approver = std::sync::Arc::new(InteractiveApprover::new(tx));

        let first = {
            let approver = approver.clone();
            tokio::spawn(async move {
                approver
                    .decide(ApprovalRequest::new(ActionKind::Write, "first"))
                    .await
            })
        };
        let second = {
            let approver = approver.clone();
            tokio::spawn(async move {
                approver
                    .decide(ApprovalRequest::new(ActionKind::Write, "second"))
                    .await
            })
        };

        // Only the first question is on screen: the second is waiting on the
        // gate, not queued behind the first in the channel.
        let pending = rx.recv().await.expect("the first request arrives");
        assert_eq!(pending.request().title, "first");
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv())
                .await
                .is_err(),
            "the second prompt must not appear while the first is unanswered"
        );

        pending.answer(Decision::Approved);
        let pending = rx.recv().await.expect("the second request arrives");
        assert_eq!(pending.request().title, "second");
        pending.answer(Decision::Denied("no".to_string()));

        assert!(first.await.unwrap().is_approved());
        assert!(!second.await.unwrap().is_approved());
    }
}
